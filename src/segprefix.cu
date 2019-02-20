#include <cstdint>
#include <cstdio>
#include <vector>

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
#define c_sat(c) (((c).flags & 0x07u) != 0x00u)
#define c_inv(c) (((c).flags & 0x3fu) == 0x38u)
#define c_has(c) (!c_sat(c) && !c_inv(c))
};

__managed__ bool formula_satisfied = false;

/************************* PREPROCESS ****************************/

__global__ void preprocess(clause *d_f1, clause *d_f2, unsigned int *d_v, int r, int log3) {
	int warp_id = (blockIdx.x << 5) + (threadIdx.x >> 5); // check
	int lane_id = threadIdx.x & 31;
	clause *formula = d_f2 + warp_id * r;
	unsigned int *valid = d_v + warp_id; // check

	for(int i = lane_id; i < r; i += 32) {
		formula[i] = d_f1[i];
	}

	__syncwarp();

	int number = warp_id;

	for(int t = 0; t < log3; ++t) {
		int tmp = number / 3;
		int branch_id = number - 3 * tmp;
		number = tmp;

		clause fc;
		clause cl;
		bool fc_found = false;
		bool invalid = false;
		unsigned int mask1 = 0xffffffffu;

		for(int i = lane_id; true; i += 32) {
			mask1 = __ballot_sync(mask1, i < r); // check for second loop

			if(!mask1) {
				break;
			}

			if(i < r) {
				cl = formula[i];
			}

			if(!fc_found) {
				int has_literals = c_has(cl); // check/improve
				int mask2 = __ballot_sync(mask1, has_literals);

				if(!mask2) {
					continue;
				}

				fc_found = true; // ZLE W KONTEKSCIE i >= r
				int *ptr_cl = (int *) &cl;
				int src_lane_id = __ffs(mask2) - 1;
				tmp = __shfl_sync(mask1, *ptr_cl, src_lane_id);
				fc = *((clause *) &tmp);

				if(fc.l[branch_id] == 0) {
					if(lane_id == 0) {
						*valid = 0;
					}

					return;
				}
			}

			for(int l = 0; l < 3; ++l) {
				for(int x = 0; x < branch_id; ++x) {
					if(!(cl.flags & (0x08u << l)) && abs8(cl.l[l]) == abs8(fc.l[x])) {
						cl.flags |= (0x08u + ((fc.l[x] & 0x80u) == 0x80u)) << l;
					}
				}

				if(cl.l[l] == fc.l[branch_id]) {
					cl.flags |= (0x08u + ((fc.l[branch_id] & 0x80u) == 0)) << l;
				}
			}

			if(__any_sync(mask1, c_inv(cl))) { // all threads within a warp need to know that!
				if(lane_id == 0) {
					*valid = 0;
				}

				return;
			}

			//printf("(warp_id: %d, lane_id: %d) substituting clause %d, flags are: 0x%02x\n", warp_id, lane_id, i + 1, cl.flags);
			formula[i] = cl;
			__syncwarp(mask1); // necessary?
		}

		if(!fc_found) { // what about the threads that were i >= r? they will continue the loop
			// whole formula satisfied! I think...
			formula_satisfied = true;
			return;
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
__global__ void sat_kernel(clause *d_f1, clause *d_f2, unsigned int *d_v, int k, int r) {
	int lane_id = threadIdx.x & 31;
	int warp_id = WARPS_NB * blockIdx.x + (threadIdx.x >> 5);
	int formula_id = warp_id / 3;
	int branch_id = warp_id - 3 * formula_id;
	unsigned int *valid = d_v + k * branch_id + formula_id;
	clause *formula = d_f1 + formula_id * r;
	clause *destination = d_f2 + (branch_id * k + formula_id) * r;
	clause fc = formula[0]; // this might be slow, use __shfl_sync()?

	// check
	if(!(fc.flags & (0x08u << branch_id))) {
		if(lane_id == 0) {
			*valid = 0;
		}

		return;
	}

	for(int i = lane_id; i < r; i += 32) {
		clause cl = formula[i];

		if(c_sat(cl)) { // sprawdzic czy jest dobrze: jak jest nullowalna, to uciekaj
			break;
		}

		for(int l = 0; l < 3; ++l) {
			for(int x = 0; x < branch_id; ++x) {
				if(!(cl.flags & (0x08u << l)) && abs8(cl.l[l]) == abs8(fc.l[x])) {
					cl.flags |= (0x08u + ((fc.l[x] & 0x80u) == 0x80u)) << l;
				}
			}

			if(cl.l[l] == fc.l[branch_id]) {
				cl.flags |= (0x08u + ((fc.l[branch_id] & 0x80u) == 0)) << l;
			}
		}

		// check
		if(__any_sync(0xffffffffu, c_inv(cl))) {
			if(lane_id == 0) {
				*valid = 0;
			}

			return;
		}

		destination[i] = cl;
	}
}

/*************************** SCAN_1D *****************************/

__device__ unsigned int id = 0;
__device__ unsigned int d_p[32];
__managed__ unsigned int nb_of_valid;

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

__global__ void scan_1d(unsigned int *d_v, int k, int range_parts, int range) {
	__shared__ unsigned int partials[33];
	__shared__ unsigned int prev;
	int tid = blockIdx.x * range + threadIdx.x;
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;

	if(tid == 0) {
		partials[0] = 0;
		prev = 0;
	}

	__syncthreads();

	for(int i = 0; i < range_parts && tid < k; tid += 1024) {
		unsigned int v = warp_scan(d_v[tid]);

		if(lane_id == 31) {
			partials[warp_id + 1] = v;
		}

		__syncthreads();

		if(warp_id == 0) {
			partials[lane_id] = warp_scan(partials[lane_id]);
		}

		__syncthreads();

		d_v[tid] = v + prev;

		__syncthreads();

		if((tid & 1023) == 1023) {
			prev = abs32(v);
		}
	}

	if((tid & 1023) == 1023) {
		d_p[blockIdx.x] = prev;
		__threadfence();

		// sprawdzic czy to zadziala <3
		if(atomicAdd(&id, 1) == gridDim.x - 1) {
			id = 0;
			d_p[lane_id] = warp_scan(d_p[lane_id]);
		}
	}
}

__global__ void propagate_1d(unsigned int *d_v, int k, int range_parts, int range) {
	__shared__ int prev;
	int tid = (blockIdx.x + 1) * range + threadIdx.x;

	if(threadIdx.x == 0) {
		prev = d_p[blockIdx.x];
	}

	__syncthreads();

	unsigned int v;

	for(int i = 0; i < range_parts && tid < k; tid += 1024) {
		v = d_v[tid] += prev;
	}

	if(tid == k + 1023) {
		nb_of_valid = v;
	}
}

/************************** SCATTER_1D ***************************/

__global__ void scatter_1d(clause *d_f1, clause *d_f2, unsigned int *d_v, int r) {
	int warp_id = (blockIdx.x << 5) + (threadIdx.x >> 5);
	unsigned int p = d_v[warp_id];
	unsigned int valid = p & 0x80000000u;
	clause *formula = d_f2 + warp_id * r;
	clause *destination = d_f1 + (valid ? abs32(p) - 1 : nb_of_valid + warp_id - abs32(p)) * r;

	for(int i = threadIdx.x & 31; i < r; i += 32) {
		destination[i] = formula[i];
	}
}

/*************************** SCAN_2D *****************************/

__inline__ __device__ unsigned int warp_scan(unsigned int v, int reminder, int lane_id) {
	for(int i = 1; i < 32; i <<= 1) {
		int _v = __shfl_up_sync(0xffffffffu, v, i);

		if(lane_id >= i && i <= reminder) { // chyba dobrze
			v += abs32(_v);
		}
	}

	return v;
}

__global__ void scan_2d(clause *d_f1, unsigned int *d_v, int k, int r, int range_parts, int range) {
	__shared__ int partials[33];
	__shared__ int prev;
	int tid = blockIdx.x * range + threadIdx.x;
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;
	int range_start = tid;
	int remainder = tid % r;
	int rem_inc = 1024 % r; // check

	if(tid == 0) {
		partials[0] = 0;
		prev = 0;
	}

	__syncthreads();

	for(int i = 0; i < range_parts && tid < k; tid += 1024) {
		clause cl = d_f1[tid];
		unsigned int satisfied = c_sat(cl) ? 0 : 0x80000001u;
		unsigned int v = warp_scan(satisfied, remainder, lane_id);

		if(lane_id == 31) {
			partials[warp_id + 1] = v;
		}

		__syncthreads();

		if(warp_id == 0) {
			partials[lane_id] = warp_scan(partials[lane_id]);
		}

		__syncthreads();

		if(tid - range_start < remainder) { // check
			d_v[tid] = v + prev;
		}

		__syncthreads();

		if((tid & 1023) == 1023) {
			prev = abs32(v);
		}

		remainder += rem_inc;

		if(remainder >= r) { // check
			remainder -= r;
		}
	}

	if((tid & 1023) == 1023) { // check?
		d_p[blockIdx.x] = prev;
		__threadfence();

		if(atomicAdd(&id, 1) == gridDim.x - 1) {
			id = 0;
			d_p[lane_id] = warp_scan(d_p[lane_id]);
		}
	}
}

__global__ void propagate_2d(unsigned int *d_v, int k, int r, int range_parts, int range) {
	// ogarnij co z tym 'k'
	__shared__ int prev;
	int tid = (blockIdx.x + 1) * range + threadIdx.x;
	int range_start = tid;
	int remainder = tid % r;
	int rem_inc = 1024 % r; // wyniesc poza funkcje?

	if(threadIdx.x == 0) {
		prev = d_p[blockIdx.x];
	}

	__syncthreads();

	for(int i = 0; i < range_parts && tid < k; tid += 1024) {
		if(tid - range_start >= remainder) { // check
			return;
		}
		
		d_v[tid] += prev;
		remainder += rem_inc;

		if(remainder >= r) {
			remainder -= r;
		}

	}
}

/************************** SCATTER_2D ***************************/

__global__ void scatter_2d(clause *d_f1, clause *d_f2, unsigned int *d_v, int r) {
	int warp_id = (blockIdx.x << 5) + (threadIdx.x >> 5);
	int shift = warp_id * r;
	unsigned int *position = d_v + shift;
	clause *formula = d_f1 + shift;
	clause *destination = d_f2 + shift;

	for(int i = threadIdx.x & 31; i < r; i += 32) {
		int p = position[i]; // check!
		unsigned int satisfied = p & 0x80000000u; // check!
		destination[satisfied ? p - 1 : nb_of_valid + warp_id - p] = formula[i];
	}
}

/************************ EXTRACT_VARS ***************************/

// from a formula, extracts variables

void extract_vars(clause *formula, int r, std::vector<bool> &assignment) {
	for(int i = 0; i < r; ++i) {
		for(int j = 0; j < 3; ++j) {
			int8_t var = formula[i].l[j];
			bool val = formula[i].flags & (0x01u << j);
			bool set = formula[i].flags & (0x08u << j);

			if(set) {
				assignment[abs8(var)] = (var < 0) ^ val; 
			}
		}
	}
}

void print_formula(clause *formula, int r) {
	for(int i = 0; i < r; ++i) {
		uint8_t *ptr = (uint8_t *) &formula[i];

		for(int j = 0; j < 4; ++j) {
			for(int k = 0; k < 8; ++k) {
				printf("%d", (ptr[j] >> 7-k) & 1);
			}

			printf(" ");
		}

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
				if((flag & 0x01u) ^ ((ptr[j] & 0x80u) >> j)) {
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

void print_batch(clause *d_f1, int nb_of_formulas, int r) {
	gpuErrchk(cudaDeviceSynchronize());
	std::vector<clause> storage(nb_of_formulas * r);
	gpuErrchk(cudaMemcpy(storage.data(), d_f1, storage.size() * sizeof(clause), cudaMemcpyDefault));
	printf("-------------------------------- BATCH -------------------------------------\n");

	for(int i = 0; i < nb_of_formulas; ++i) {
		printf("----- FORMULA %d/%d -----\n", i + 1, storage.size());
		print_formula(storage.data() + i * r, r);
	}

	printf("\n");
}

/**************************** SWAP *******************************/

void swap(std::vector<clause> &storage, clause *d_f1, clause *d_f2, int nb_of_formulas, int s, int r) {
	int surplus = nb_of_formulas - s/3;

	if(surplus > 0) {
		int transfer = surplus * r;
		storage.resize(storage.size() + transfer);
		clause *dst = storage.data() + storage.size() - transfer;
		clause *src = d_f1 + nb_of_formulas * r - transfer;
		gpuErrchk(cudaMemcpy(dst, src, transfer * sizeof(clause), cudaMemcpyDefault));
		nb_of_formulas -= surplus;
	}

	if(surplus < 0) {
		int transfer = -surplus * r;
		clause *dst = d_f1 + nb_of_formulas * r;
		clause *src = storage.data() + storage.size() - transfer;
		gpuErrchk(cudaMemcpy(dst, src, transfer * sizeof(clause), cudaMemcpyDefault));
		storage.resize(storage.size() - transfer);
		nb_of_formulas += -surplus;
	}
}

/************************** PIPELINE *****************************/

void pipeline(std::vector<clause> &storage, int n, int r, int s, int log3) {
	int nb_of_formulas = s;
	clause *d_f1;
	clause *d_f2;
	unsigned int *d_v;
	gpuErrchk(cudaMalloc(&d_f1, s * r * sizeof(clause)));
	gpuErrchk(cudaMalloc(&d_f2, s * r * sizeof(clause)));
	gpuErrchk(cudaMalloc(&d_v, s * sizeof(unsigned int)));
	gpuErrchk(cudaMemcpy(d_f1, storage.data(), r * sizeof(clause), cudaMemcpyDefault));
	storage.resize(0);

	// spawns ceil(s/32) warps, each warp generating a singe new formula
	preprocess<<<(nb_of_formulas + 31)/32, 1024>>>(d_f1, d_f2, d_v, r, log3);
	print_batch(d_f2, nb_of_formulas, r);
	return; // --------------------------------

	// czy powinna zwrocic prawdziwe 's'?

	while(nb_of_formulas) { // check
		std::swap(d_f1, d_f2);
		// jak ma sie to blokowanie do warunku w while-u?
		gpuErrchk(cudaDeviceSynchronize());

		if(formula_satisfied) {
			// DO SOMETHING
			return;
		}

		int range_parts = (nb_of_formulas + 32 * 1024 - 1) / (32 * 1024); // check
		int range = range_parts * 1024;
		int blocks = (s + range - 1) / range;
		scan_1d<<<blocks, 1024>>>(d_v, s, range_parts, range);

		if(blocks > 1) {
			propagate_1d<<<blocks - 1, 1024>>>(d_v, nb_of_formulas, range_parts, range); //check
		}

		scatter_1d<<<(nb_of_formulas + 31) / 32, 1024>>>(d_f1, d_f2, d_v, r);
		gpuErrchk(cudaDeviceSynchronize());

		nb_of_formulas = nb_of_valid;
		range_parts = (r * nb_of_formulas + 32 * 1024 - 1) / (32 * 1024); // check
		range = range_parts * 1024;
		blocks = (s + range - 1) / range;

		scan_2d<<<blocks, 1024>>>(d_f1, d_v, nb_of_formulas, r * nb_of_formulas, range_parts, range); // check

		if(blocks > 1) {
			propagate_2d<<<blocks - 1, 1024>>>(d_v, nb_of_formulas, r, range_parts, range); // check order
		}

		scatter_2d<<<(nb_of_formulas + 31) / 32, 1024>>>(d_f1, d_f2, d_v, r);
		swap(storage, d_f1, d_f2, nb_of_formulas, s, r);
		sat_kernel<<<(nb_of_formulas + 31) / 32, 32 * WARPS_NB>>>(d_f1, d_f2, d_v, nb_of_formulas, r);
		// czy nie blokujemy?
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
	int log3 = 0;

	while(3 * s <= BATCH_SIZE) {
		s *= 3;
		++log3;
	}

	std::vector<clause> formulas(BATCH_SIZE * r);

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
			} else {
				formulas[i].l[j] = (int8_t) -var | 0x80u;
			}

			++j;
		}

		while(j < 3) {
			formulas[i].flags |= 0x8u << j;
			++j;
		}
	}

	print_formula(formulas.data(), r);
	pipeline(formulas, n, r, s, log3);

}
