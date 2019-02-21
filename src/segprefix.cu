#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define BATCH_SIZE 27
#define WARPS_NB 10
#define abs8(n) ((n) & 0x7fu)
#define abs32(n) ((n) & 0x7fffffffu)

struct clause {
	/* Field 'flags':
	 *   0x01u - value of literal l[0] (taking into account the sign)
	 *   0x02u - value of literal l[1] (taking into account the sign)
	 *   0x04u - value of literal l[2] (taking into account the sign)
	 *   0x08u - literal l[0] was assigned a value
	 *   0x10u - literal l[1] was assigned a value
	 *   0x20u - literal l[2] was assigned a value
	 * Satisfied if: (flags & 0x07u) != 0x00u
	 * Invalid if:   (flags & 0x3fu) == 0x2au
	 * Has literals: not satisfied and not invalid
	 */
	uint8_t l[3];
	uint8_t flags;
//
#define c_sat(c) (((c).flags & 0x07u) != 0x00u)

// 
#define c_inv(c) (((c).flags & 0x3fu) == 0x38u)

// there is any reason to substitute any literal to get the formula satisfied
#define c_has(c) (!c_sat(c) && !c_inv(c))
};

__managed__ int formula_satisfied = -1; // consider making a position of a satisfied formula

/************************* PREPROCESS ****************************/

__global__ void preprocess(clause *d_f1, clause *d_f2, unsigned int *d_v, int r, int log3r, int nb_of_formulas) {
	int warp_id = (blockIdx.x << 5) + (threadIdx.x >> 5);

	if(warp_id >= nb_of_formulas) {
		return;
	}

	int lane_id = threadIdx.x & 31;
	int number = warp_id;
	clause *formula = d_f2 + warp_id * r;
	unsigned int *valid = d_v + warp_id; // check

	for(int i = lane_id; i < r; i += 32) {
		formula[i] = d_f1[i];
	}

	__syncwarp();


	for(int t = 0; t < log3r; ++t) {
		int tmp = number / 3;
		int branch_id = number - 3 * tmp;
		number = tmp;

		clause fc;
		bool fc_found = false;

		for(int i = lane_id; !fc_found && __any_sync(0xffffffffu, i < r); i += 32) { // bez sync
			int mask = __ballot_sync(0xffffffffu, i < r ? c_has(fc = formula[i]) : 0);

			if(!mask) {
				continue;
			}

			fc_found = true;
			int *ptr_fc = (int *) &fc;
			int src_lane_id = __ffs(mask) - 1;
			tmp = __shfl_sync(0xffffffffu, *ptr_fc, src_lane_id);
			fc = *((clause *) &tmp);

			if(fc.flags & (0x08u << branch_id)) { // chyba OK
				if(lane_id == 0) {
					*valid = 1;
				}

				return;
			}
		}

		if(!fc_found) {
			if(lane_id == 0) {
				atomicExch(&formula_satisfied, warp_id); // check
			}

			return;
		}

		for(int i = lane_id; __any_sync(0xfffffffu, i < r); i += 32) { // da sie bez synca
			clause cl;

			if(i < r) {
				cl = formula[i];

				for(int l = 0; l < 3; ++l) {
					for(int x = 0; x < branch_id; ++x) {
						if(abs8(cl.l[l]) == abs8(fc.l[x])) { // CHECK!!!!!!!!!!!!!!!!!!!!!
							bool fc_neg = fc.l[x] & 0x80u;
							bool cl_neg = cl.l[l] & 0x80u;
							cl.flags |= (0x09u - (fc_neg == cl_neg)) << l;
						}
					}

					if(abs8(cl.l[l]) == abs8(fc.l[branch_id])) {
						bool fc_neg = fc.l[branch_id] & 0x80u;
						bool cl_neg = cl.l[l] & 0x80u;
						cl.flags |= (0x08u + (fc_neg == cl_neg)) << l;
					}
				}
			}

			if(__any_sync(0xffffffffu, i < r ? c_inv(cl) : 0)) { // check
				if(lane_id == 0) {
					*valid = 1; // check
				}

				return;
			}

			if(i < r) {
				formula[i] = cl;
			}

			__syncwarp();
		}
	}
}

/************************* SAT_KERNEL ****************************/

/* Triples the number of formulas in a batch and marks invalid/missing ones
 * 
 * d_f - array of formulas and a free space for new formulas
 * d_v - array of flags indicating whether a formula is valid or not
 * k - number of formulas to triple
 * r - total number of clauses
 */
__global__ void sat_kernel(clause *d_f1, clause *d_f2, unsigned int *d_v, int b, int r) {
	int warp_id = WARPS_NB * blockIdx.x + (threadIdx.x >> 5);

	if(warp_id >= b) { // check
		return;
	}

	int lane_id = threadIdx.x & 31;
	int formula_id = warp_id / 3;
	int branch_id = warp_id - 3 * formula_id;
	unsigned int *valid = d_v + b * branch_id + formula_id;
	clause *formula = d_f1 + formula_id * r;
	clause *destination = d_f2 + (branch_id * b + formula_id) * r;
	clause fc = formula[0]; // this might be slow, use __shfl_sync()?

	if(fc.flags & (0x08u << branch_id)) { // CHECK!!!!!!!!!!!!!!
		if(lane_id == 0) {
			*valid = 1;
		}

		return;
	}

	bool satisfied = true; // CHECK

	for(int i = lane_id; __any_sync(0xffffffffu, i < r); i += 32) {
		clause cl;

		if(i < r) {
			cl = formula[i];

			for(int l = 0; l < 3; ++l) {
				for(int x = 0; x < branch_id; ++x) {
					if(abs8(cl.l[l]) == abs8(fc.l[x])) { // CHECK!!!!!!!!!!!!!!!!!!!!!
						bool fc_neg = fc.l[x] & 0x80u;
						bool cl_neg = cl.l[l] & 0x80u;
						cl.flags |= (0x09u - (fc_neg == cl_neg)) << l;
					}
				}

				if(abs8(cl.l[l]) == abs8(fc.l[branch_id])) {
					bool fc_neg = fc.l[branch_id] & 0x80u;
					bool cl_neg = cl.l[l] & 0x80u;
					cl.flags |= (0x08u + (fc_neg == cl_neg)) << l;
				}
			}

			if(!c_sat(cl)) {
				satisfied = false;
			}
		}

		if(__any_sync(0xffffffffu, i < r ? c_inv(cl) : 0)) { // check
			if(lane_id == 0) {
				*valid = 1;
			}

			return;
		}

		if(i < r) {
		destination[i] = cl;
		}

		if(__any_sync(0xffffffffu, i < r ? c_sat(cl) : 1)) { // check
			break;
		}

	}

	if(__all_sync(0xffffffffu, satisfied)) {
		if(lane_id == 0) {
			formula_satisfied = warp_id;
		}
	}
}

/*************************** SCAN_1D *****************************/

__device__ unsigned int id = 0;
__device__ unsigned int d_p[32];
__managed__ int nb_of_valid;

__inline__ __device__ unsigned int warp_scan(unsigned int v) {
	int lane_id = threadIdx.x & 31;

	for(int i = 1; i < 32; i <<= 1) {
		int _v = __shfl_up_sync(0xffffffffu, v, i);

		if(lane_id >= i) {
			v += abs32(_v);
		}
	}

	return v;
}

__global__ void scan_1d(unsigned int *d_v, int b, int range_parts, int range) {
	__shared__ unsigned int partials[33];
	__shared__ unsigned int prev;
	int tid = blockIdx.x * range + threadIdx.x;
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;

	if(threadIdx.x == 0) {
		partials[0] = 0;
		prev = 0;
	}

	__syncthreads();

	for(int i = 0; i < range_parts && tid < b; ++i, tid += 1024) {
		unsigned int v = warp_scan(d_v[tid] ? 0 : 0x80000001u);

		if(lane_id == 31) {
			partials[warp_id + 1] = v;
		}

		__syncthreads();

		if(warp_id == 0) {
			partials[lane_id] = warp_scan(partials[lane_id]);
		}

		__syncthreads();

		// DO JASNEJ CHOLERY, DODAJ PARTIALE!!! a teraz checkuj to!
		d_v[tid] = v + partials[warp_id] + prev;

		__syncthreads();

		if((tid & 1023) == 1023) {
			prev = abs32(v);
		}
	}

	if((tid & 1023) == 1023) {
		d_p[blockIdx.x] = prev;
	}

	/*
	if((tid & 1023) == 1023) {
		d_p[blockIdx.x] = prev;
		__threadfence();

		// sprawdzic czy to zadziala <3
		if(atomicAdd(&id, 1) == gridDim.x - 1) {
			id = 0;
			d_p[lane_id] = warp_scan(d_p[lane_id]);
		}
	}
	*/
}

__global__ void small_scan_1d() {
	int lane_id = threadIdx.x & 31;
	d_p[lane_id] = warp_scan(d_p[lane_id]);
}

__global__ void propagate_1d(unsigned int *d_v, int b, int range_parts, int range) {
	__shared__ int prev;
	int tid = (blockIdx.x + 1) * range + threadIdx.x;

	if(threadIdx.x == 0) {
		prev = abs32(d_p[blockIdx.x]); // check
	}

	__syncthreads();

	for(int i = 0; i < range_parts && tid < b; tid += 1024) {
		d_v[tid] += prev;
	}
}

/************************** SCATTER_1D ***************************/

__global__ void scatter_1d(clause *d_f1, clause *d_f2, unsigned int *d_v, int r, int b) {
	// IF INVALID THEN WHY THE HECK YOU SORT THEM!?!?!?!?!
	__shared__ unsigned int values;
	int warp_id = (blockIdx.x << 5) + (threadIdx.x >> 5);

	if(warp_id >= b) {
		return;
	}

	if(threadIdx.x == 0) {
		values = abs32(d_v[b - 1]);

		if(blockIdx.x == 0) {
			nb_of_valid = values;
		}
	}

	__syncthreads();

	unsigned int value = d_v[warp_id];

	if(!(value & 0x80000000u)) {
		return;
	}

	int pos = abs32(value);
	clause *formula = d_f1 + warp_id * r;
	clause *destination = d_f2 + (pos - 1) * r;

	for(int i = threadIdx.x & 31; i < r; i += 32) {
		destination[i] = formula[i];
	}
}

/*************************** SCAN_2D *****************************/

/* If id is the number of a position, then it adds all elements
 * from 0 to id%r positions behind
 */
__inline__ __device__ unsigned int warp_scan(unsigned int v, int reminder, int lane_id) {
	for(int i = 1; i < 32; i <<= 1) {
		int _v = __shfl_up_sync(0xffffffffu, v, i);

		if(lane_id >= i && i <= reminder) {
			v += abs32(_v);
		}
	}

	return v;
}

__device__ void helper_print_partials(int *partials) {
	printf("Printing partials from (blockIdx.x: %d, threadIdx.x: %d)\n", blockIdx.x, threadIdx.x);

	for(int i = 0; i < 32; ++i) {
		printf("partials[%d]: %d\n", i, partials[i]);
	}

	printf("\n");
}

__global__ void scan_2d(clause *d_f2, unsigned int *d_v, int b, int r, int range_parts, int range) {
	__shared__ int partials[33];
	__shared__ int prev;
	int range_start = blockIdx.x * range; // check
	int tid = range_start + threadIdx.x; // check
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;
	int remainder = tid % r;
	int rem_inc = 1024 % r; // check

	if(threadIdx.x == 0) {
		partials[0] = 0;
		prev = 0;
	}

	__syncthreads();

	for(int i = 0; i < range_parts && tid < b; ++i, tid += 1024, range_start += 1024) {
		clause cl = d_f2[tid];
		unsigned int satisfied = c_sat(cl) ? 0 : 0x80000001u; // chyba tylko 0 lub 1 wystarczy
		unsigned int v = warp_scan(satisfied, remainder, lane_id);

		if(lane_id == 31) {
			partials[warp_id + 1] = v;
		}

		__syncthreads();

		if(warp_id == 0) {
			if(blockIdx.x == 1 && lane_id == 0) {
				helper_print_partials(partials);
			}

			int t = partials[lane_id];

			for(int u = 1; u < 32; u <<= 1) {
				int _t = __shfl_up_sync(0xffffffffu, t, u);

				if(lane_id >= u && u*32 <= (range_start + 32 * lane_id) % r) {
					t += abs32(_t);
				}
			}

			partials[lane_id] = abs32(t);
		}

		__syncthreads();

		if(tid - range_start - 32 * warp_id < remainder) {
			v += abs32(partials[warp_id]);
		}

		if(tid - range_start < remainder) { // chyba <
			v += abs32(prev);
		}

		d_v[tid] = v;

		__syncthreads();

		if((tid & 1023) == 1023) {
			prev = abs32(v);
		}

		remainder += rem_inc;

		if(remainder >= r) {
			remainder -= r;
		}
	}

	if((tid & 1023) == 1023) {
		d_p[blockIdx.x] = prev;
	}
}

__global__ void small_scan_2d(int range, int r) {
	int lane_id = threadIdx.x & 31;
	unsigned int v = d_p[lane_id];

	for(int i = 1; i < 32; i <<= 1) {
		int _v = __shfl_up_sync(0xffffffffu, v, i);

		if(lane_id >= i && i * range <= (range * (lane_id + 1)) % r) {
			v += abs32(_v);
		}
	}

	d_p[lane_id] = v;
}

__global__ void propagate_2d(unsigned int *d_v, int b, int r, int range_parts, int range) {
	__shared__ int prev;
	int range_start = (blockIdx.x + 1) * range;
	int tid = range_start + threadIdx.x;
	int remainder = tid % r;
	int rem_inc = 1024 % r;

	if(threadIdx.x == 0) {
		prev = abs32(d_p[blockIdx.x]);
	}

	__syncthreads();

	for(int i = 0; i < range_parts && tid < b && tid - range_start < remainder; tid += 1024) {
		d_v[tid] += prev;
		remainder += rem_inc;

		if(remainder >= r) {
			remainder -= r;
		}
	}
}

/************************** SCATTER_2D ***************************/

__global__ void scatter_2d(clause *d_f1, clause *d_f2, unsigned int *d_v, int r, int b) {
	int warp_id = (blockIdx.x << 5) + (threadIdx.x >> 5);

	if(warp_id >= b) {
		return;
	}

	int lane_id = threadIdx.x & 31;
	int shift = warp_id * r;
	unsigned int *pos = d_v + shift;
	clause *formula = d_f2 + shift;
	clause *destination = d_f1 + shift;
	int nb_not_null;

	if(lane_id == 0) {
		nb_not_null = abs32(pos[r - 1]);
	}

	nb_not_null = __shfl_sync(0xffffffffu, nb_not_null, 0);

	for(int i = threadIdx.x & 31; i < r; i += 32) {
		unsigned int value = pos[i];
		bool not_null = value & 0x80000000u;
		int new_pos = not_null ? abs32(value) - 1 : nb_not_null + i - abs32(value);
		destination[new_pos] = formula[i];
	}
}

/************************ EXTRACT_VARS ***************************/

// from a formula, extracts variables

void extract_vars(clause *formula, int r, std::vector<bool> &assignments, int n) {
	for(int i = 0; i < r; ++i) {
		for(int j = 0; j < 3; ++j) {
			uint8_t var = formula[i].l[j];
			bool sign = ((var & 0x80u) >> 7);
			bool value = (formula[i].flags & (0x01u << j)) >> j;
			assignments[abs8(var) - 1] = sign ^ value; 
		}
	}
}

void print_formula(clause *formula, int r) {
	for(int i = 0; i < r; ++i) {
		uint8_t *ptr = (uint8_t *) &formula[i];

		/*
		for(int j = 0; j < 4; ++j) {
			for(int k = 0; k < 8; ++k) {
				printf("%d", (ptr[j] >> 7-k) & 1);
			}

			printf(" ");
		}
		*/

		printf("\t");

		for(int j = 0; j < 3; ++j) {
			if(ptr[j] & 0x80u) {
				printf("%d", -abs8(ptr[j]));
			} else {
				printf("%d", ptr[j]);
			}

			printf("\t");
		}

		for(int j = 0; j < 3; ++j) {
			uint8_t flag = (ptr[3] & (0x09u << j)) >> j;

			if(ptr[j] != 0 && (flag & 0x08u)) {
				if((flag & 0x01u) /*^ ((ptr[j] & 0x80u) >> j)*/) {
					printf("true");
				} else {
					printf("false");
				}
			} else {
				printf("X");
			}

			if(j != 2) {
				printf("\t");
			}
		}

		printf("\n");
	}

	printf("\n");
}

void print_batch(clause *d_f1, unsigned int *d_v, int nb_of_formulas, int r) {
	gpuErrchk(cudaDeviceSynchronize());

	std::vector<clause> storage(nb_of_formulas * r);
	std::vector<unsigned int> validities(nb_of_formulas);
	gpuErrchk(cudaMemcpy(storage.data(), d_f1, storage.size() * sizeof(clause), cudaMemcpyDefault));

	printf("-------------------------------- BATCH -------------------------------------\n");
	printf("Formula is %s\n\n", formula_satisfied != -1 ? "satisfied" : "unsatisfied");

	for(int i = 0; i < nb_of_formulas; ++i) {
		printf("----- FORMULA %d/%d -----\n", i + 1, nb_of_formulas);
		print_formula(storage.data() + i * r, r);
	}

	printf("\n");
	gpuErrchk(cudaDeviceSynchronize());
}

/**************************** SWAP *******************************/

void swap(std::vector<clause> &storage, clause *d_f1, int nb_of_formulas, int s, int r) {
	int surplus = nb_of_formulas - s/3;

	if(surplus > 0) {
		int transfer = surplus * r;
		storage.resize(storage.size() + transfer);
		clause *dst = storage.data() + storage.size() - transfer;
		clause *src = d_f1 + nb_of_formulas * r - transfer;
		gpuErrchk(cudaMemcpy(dst, src, transfer * sizeof(clause), cudaMemcpyDefault));
		nb_of_formulas -= surplus;
	} else if(surplus < 0) {
		int transfer = -surplus * r;
		clause *dst = d_f1 + nb_of_formulas * r;
		clause *src = storage.data() + storage.size() - transfer;
		gpuErrchk(cudaMemcpy(dst, src, transfer * sizeof(clause), cudaMemcpyDefault));
		storage.resize(storage.size() - transfer);
		nb_of_formulas += -surplus;
	}
}

void print_d_v(unsigned int *d_v, int b, int r) {
	gpuErrchk(cudaDeviceSynchronize());

	std::vector<unsigned int> v(b * r);
	gpuErrchk(cudaMemcpy(v.data(), d_v, b * r * sizeof(unsigned int), cudaMemcpyDefault));

	printf("================ PRINTING d_v ================\n");

	for(int i = 0; i < b; ++i) {
		for(int j = 0; j < r; ++j) {
			int id = i * r + j;
			unsigned int value = v[id];
			printf("d_v[%d]: %d-%u\n", id, !!(value & 0x80000000u), abs32(value));
		}

		printf("\n");
	}

	printf("-----------------\n\n");

	gpuErrchk(cudaDeviceSynchronize());
}

/************************** PIPELINE *****************************/

void pipeline(std::vector<clause> &storage, int n, int r, int s, int log3r, std::vector<bool> &assignments) {
	int nb_of_formulas = s;
	clause *d_f1;
	clause *d_f2;
	unsigned int *d_v;
	gpuErrchk(cudaMalloc(&d_f1, s * r * sizeof(clause)));
	gpuErrchk(cudaMalloc(&d_f2, s * r * sizeof(clause)));
	gpuErrchk(cudaMalloc(&d_v, s * r * sizeof(unsigned int)));
	gpuErrchk(cudaMemcpy(d_f1, storage.data(), r * sizeof(clause), cudaMemcpyDefault));
	gpuErrchk(cudaMemset(d_v, 0, nb_of_formulas * sizeof(unsigned int)));
	storage.resize(0);

	preprocess<<<(nb_of_formulas + 31)/32, 1024>>>(d_f1, d_f2, d_v, r, log3r, nb_of_formulas);

	while(nb_of_formulas) { // check
		std::swap(d_f1, d_f2);
		// jak ma sie to blokowanie do warunku w while-u?
		gpuErrchk(cudaDeviceSynchronize());

		if(formula_satisfied != -1) {
			printf("LEAVING THAT HELL!\n");
			gpuErrchk(cudaMemcpy(storage.data(), d_f1 + formula_satisfied * r, r * sizeof(clause), cudaMemcpyDefault));
			extract_vars(storage.data(), r, assignments, n);
			return;
		}

		int range_parts = (nb_of_formulas + 32 * 1024 - 1) / (32 * 1024); // check
		int range = range_parts * 1024;
		int blocks = (nb_of_formulas + range - 1) / range;

		printf("SCAN 1D\n"); //
		printf("nb_of_formulas: %d\n", nb_of_formulas); //
		printf("range_parts: %d\n", range_parts); //
		printf("range: %d\n", range); //
		printf("blocks: %d\n", blocks); //
		printf("\n"); //

		scan_1d<<<blocks, 1024>>>(d_v, s, range_parts, range);

		if(blocks > 1) {
			printf("blocks > 1\n"); //
			small_scan_1d<<<1, 32>>>();
			propagate_1d<<<blocks - 1, 1024>>>(d_v, nb_of_formulas, range_parts, range); //check
		}

		scatter_1d<<<(nb_of_formulas + 31) / 32, 1024>>>(d_f1, d_f2, d_v, r, nb_of_formulas);
		gpuErrchk(cudaDeviceSynchronize());

		nb_of_formulas = nb_of_valid;
		range_parts = (r * nb_of_formulas + 32 * 1024 - 1) / (32 * 1024); // check
		range = range_parts * 1024;
		blocks = (r * nb_of_formulas + range - 1) / range;

		printf("SCAN 2D\n"); //
		printf("nb_of_formulas: %d\n", nb_of_formulas); //
		printf("range_parts: %d\n", range_parts); //
		printf("range: %d\n", range); //
		printf("blocks: %d\n", blocks); //
		printf("\n"); //

		//printf("Before scan_2d\n");
		//print_d_v(d_v, nb_of_formulas, r);

		//////// CHECK ARGUMENTS!
		scan_2d<<<blocks, 1024>>>(d_f2, d_v, r * nb_of_formulas, r, range_parts, range); // check
		gpuErrchk(cudaDeviceSynchronize());
		//printf("After scan_2d\n");
		//print_d_v(d_v, nb_of_formulas, r);

		if(blocks > 1) {
			small_scan_2d<<<1, 32>>>(range, r);
			gpuErrchk(cudaDeviceSynchronize());
			//printf("After small_scan_2d\n");
			//print_d_v(d_v, nb_of_formulas, r);
			propagate_2d<<<blocks - 1, 1024>>>(d_v, r * nb_of_formulas, r, range_parts, range);
			gpuErrchk(cudaDeviceSynchronize());
			printf("After propagate_2d\n");
			print_d_v(d_v, nb_of_formulas, r);
		}

		scatter_2d<<<(nb_of_formulas + 31) / 32, 1024>>>(d_f1, d_f2, d_v, r, nb_of_formulas);
		//print_batch(d_f2, d_v, nb_of_formulas, r); //

		//gpuErrchk(cudaDeviceSynchronize());
		//printf("After scatter_2d\n");
		//print_d_v(d_v, nb_of_formulas, r);

		gpuErrchk(cudaDeviceSynchronize()); // ma byc?
		swap(storage, d_f1, nb_of_formulas, s, r);
		return; // 

		// czy nie blokujemy?
		//gpuErrchk(cudaMemset(d_v, 0, 3 * nb_of_formulas * sizeof(unsigned int))); // CHECK!!!

		// sat_kernel<<<(nb_of_formulas + 31) / 32, 32 * WARPS_NB>>>(d_f1, d_f2, d_v, nb_of_formulas, r); // CHECK CHECK CHECK
	}

	cudaFree(d_f1);
	cudaFree(d_f2);
	cudaFree(d_v);
}

/**************************** MAIN *******************************/

int main() {
	int n, r;
	scanf("%d %d", &n, &r);
	int s = 1;
	int log3r = 0;

	while(3 * s <= BATCH_SIZE) {
		s *= 3;
		++log3r;
	}

	std::vector<clause> formulas(BATCH_SIZE * r);
	std::vector<int> freq(n, 0);
	std::vector<bool> assignments;

	for(int i = 0; i < r; ++i) {
		int j = 0;

		while(j < 4) {
			int var;
			scanf("%d", &var);

			if(j == 3 || var == 0) {
				break;
			}

			if(var > 0) {
				formulas[i].l[j] = (int8_t) var;
				++freq[var];
			} else {
				formulas[i].l[j] = (int8_t) -var | 0x80u;
				++freq[-var];
			}

			++j;
		}

		while(j < 3) {
			formulas[i].flags |= 0x08u << j;
			++j;
		}
	}

	std::sort(formulas.begin(), formulas.begin() + r,
			[&freq](clause &a, clause &b) -> bool {
			int sum1 = abs8(a.l[0]) + abs8(a.l[1]) + abs8(a.l[2]);
			int sum2 = abs8(b.l[0]) + abs8(b.l[1]) + abs8(b.l[2]);
			return sum1 > sum2;
			});

	pipeline(formulas, n, r, s, log3r, assignments);
}

// TODO: generalize 1024 to blockDim.x
// TODO: change name of variable 'valid' to 'invalid'
// TODO: extract the function that assigns a clause based on another clause/values
// TODO: 1d scan: remove separate summation by making some iffs and letting all threads come across the __syncthreads()
// TODO: 1d ccan works only because range is divisible by 1024
