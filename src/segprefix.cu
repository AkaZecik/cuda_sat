#include <cstdio>
#include <stdint.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define WARPS_NB 10
#define abs(n) ((n) & 0x7fu)

struct clause {
	/* Field 'flags':
	 *   0x01u - value of literal l[0]
	 *   0x02u - value of literal l[1]
	 *   0x04u - value of literal l[2]
	 *   0x08u - literal l[0] was assigned
	 *   0x10u - literal l[1] was assigned
	 *   0x20u - literal l[2] was assigned
	 * Nullified if: !((c.flags & 0x2au) && (c.flags & 0x15u))
	 * Invalid if: (c.flags & 0x3fu) == 0x2au
	 */
	int8_t l[3];
	uint8_t flags;
};

/************************* SAT_KERNEL ****************************/

/* Triples the number of formulas in a batch and marks invalid/missing ones
 * 
 * d_f - array of formulas and a free space for new formulas
 * d_v - array of flags indicating whether a formula is valid or not
 * k - number of formulas to triple
 * r - total number of clauses
 */
__global__ void sat_kernel(clause *const d_f, int *const d_v, int k, int r) {
	int warp_id = WARPS_NB * blockIdx.x + (threadIdx.x >> 5);
	int formula_id = warp_id / 3;
	int branch_id = warp_id - 3 * formula_id;
	int *valid = d_v + k * branch_id + formula_id;
	clause *formula = d_f + formula_id * r;
	clause *destination = d_f + (3 * k * branch_id + formula_id) * r;
	clause fc = formula[0];

	if(!(fc.flags & (0x08u << branch_id))) {
		*valid = 0;
		return;
	}

	for(int i = threadIdx.x & 31; i < r; i += 32) {
		clause cl = formula[i];

		for(int l = 0; l < 3; ++l) {
			for(int y = 0; y < branch_id; ++y) {
				if(!(cl.flags & (0x08u << l)) && abs(cl.l[l]) == abs(fc.l[y])) {
					cl.flags |= (0x08u + (fc.l[y] < 0)) << l;
				}
			}

			if(cl.l[l] == fc.l[branch_id]) {
				cl.flags |= (0x08u + (fc.l[branch_id] > 0)) << l;
			}
		}

		if((cl.flags & 0x3fu) == 0x2au) {
			*valid = 0;
		}

		destination[i] = cl;
	}
}

/*************************** 1D_SCAN *****************************/

__device__ volatile int id = 0;
__device__ volatile int d_p[32];

__inline__ __device__ int warp_scan(int v) {
	int lane_id = threadIdx.x & 31;

	for(int i = 1; i < 32; i <<= 1) {
		int _v = __shfl_up_sync(0xffffffffu, v, i);

		if(lane_id >= i) {
			v += _v;
		}
	}

	return v;
}

__global__ void 1d_scan(int *d_v, int k, int range_parts) {
	__shared__ partials[33];
	__shared__ prev;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;

	if(tid == 0) {
		values[0] = 0;
		prev = 0;
	}

	__syncthreads();

	for(int i = 0; i < range_parts && tid < k; tid += 1024) {
		int v = warp_scan(d_v[tid]);

		if(lane_id == 31) {
			values[warp_id + 1] = v;
		}

		__syncthreads();

		if(warp_id == 0) {
			partials[lane_id] = warp_scan(partials[lane_id]);
		}

		__syncthreads();

		d_v[tid] = p + prev;

		__syncthreads();

		if((tid & 1023) == 1023) {
			prev = p;
		}
	}

	if((tid & 1023) == 1023) {
		d_p[blockIdx.x] = p;
		__threadfence();

		if(atomicAdd(&id, 1) == gridDim.x - 1) {
			id = 0;
			d_p[lane_id] = warp_scan(d_p[lane_id]);
		}
	}
}

__global__ void 1d_propagate(int *d_v, int k, int range_parts) {
	
}

/************************** 1D_DEFRAG ****************************/

__global__ void 1d_defrag(clause *d_f, int *d_v, int k, int r) {
	int warp_id = (blockIdx.x << 5) + (threadIdx.x >> 5);
	int formula_size = r * sizeof(clause);
	int shift = 3 * k * formula_size;
	int new_position = d_v[warp_id] * formula_size; // might not work

	for(int i = threadIdx.x & 31; i < r; i += 32) {
		if(flag is 1) { // fix it
			d_f[new_position + i] = d_f[shift + warp_id + i];
		}
	}
}

/*************************** 2D_SCAN *****************************/

__global__ void 2d_scan(clause *d_f, int *d_v, int k, int r) {

}

/************************** 2D_DEFRAG ****************************/

__global__ void 2d_defrag(clause *d_f, int *d_v, ) {

}

/**************************** SWAP *******************************/

void pipeline() {
	while(true) {
		1d_scan<<<0, 0>>>();
		1d_defrag<<<0, 1024>>>();
		2d_scan<<<0, 1024>>>();
		2d_defrag<<<0, 1024>>>();
		swap();
		sat_kernel<<<0, 0>>>();
	}
}

/**************************** MAIN *******************************/

int main() {
	int n;
	int r;
	int s;
}

////////////////////////////////////////////////////////////////////////

struct pair {
	int v;
	int g;
};

__inline__ __device__ pair warp_prefix_scan(int v, int g) {
	int lane_id = threadIdx.x & 31;

	for(int i = 1; i < 32; i *= 2) {
		int _v = __shfl_up_sync(-1, v, i);
		int _g = __shfl_up_sync(-1, g, i);

		if(lane_id >= i) {
			v = g ? v : _v + v;
			g |= _g;
		}
	}

	return {v, g};
}

__device__ int d_v[32];
__device__ int d_g[32];

__global__ void block_prefix_scan(int *d_values, int *d_groups, int n, int parts, int range) {
	__shared__ int values[33];
	__shared__ int groups[33];
	__shared__ int prev_v;
	__shared__ int prev_g;

	if(threadIdx.x == 0) {
		values[0] = 0;
		groups[0] = 0;
		prev_v = 0;
		prev_g = 0;
	}

	__syncthreads();

	int tid = blockIdx.x * range + threadIdx.x;
	int warp_id = threadIdx.x >> 5;
	int lane_id = threadIdx.x & 31;
	int i;

	for(i = 0; i < parts; ++i, tid += 1024) {
		if(tid >= n) {
			return;
		}

		pair p(warp_prefix_scan(d_values[tid], d_groups[tid]));

		if(lane_id == 31) {
			values[warp_id + 1] = p.v;
			groups[warp_id + 1] = p.g;
		}

		__syncthreads();

		if(warp_id == 0) {
			pair q(warp_prefix_scan(values[lane_id], groups[lane_id]));
			values[lane_id] = q.v;
			groups[lane_id] = q.g;
		}

		__syncthreads();

		if(p.g == 0) {
			p.v += values[warp_id];
			p.g |= groups[warp_id];
		}

		if(p.g == 0) {
			p.v += prev_v;
			p.g |= prev_g;
		}

		d_values[tid] = p.v;
		d_groups[tid] = p.g;

		__syncthreads();

		if((tid & 1023) == 1023) {
			prev_v = p.v;
			prev_g = p.g;
		}
	}

	if(i == parts && (tid & 1023) == 1023) {
		d_v[blockIdx.x] = prev_v;
		d_g[blockIdx.x] = prev_g;
	}
}

__global__ void partials() {
	pair p(warp_prefix_scan(d_v[threadIdx.x], d_g[threadIdx.x]));
	d_v[threadIdx.x] = p.v;
	d_g[threadIdx.x] = p.g;
}

__global__ void propagate(int *d_values, int *d_groups, int n, int parts, int range) {
	__shared__ int prev_v;
	int tid = range * (blockIdx.x + 1) + threadIdx.x;

	if(threadIdx.x == 0) {
		prev_v = d_v[blockIdx.x];
	}

	__syncthreads();

	for(int i = 0; i < parts; ++i, tid += 1024) {
		if(tid >= n || d_groups[tid] != 0) {
			return;
		}

		d_values[tid] += prev_v;
	}
}

__global__ void fill_groups(int *d_groups, int *d_groups_original, int m) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < m - 1) {
		d_groups[d_groups_original[tid] + 1] = 1;

		if(tid == 0) {
			d_groups[0] = 1;
		}
	}
}

void segmentedPrefixSum(std::vector<int>& values, const std::vector<int>& groups) {
	int n = values.size();
	int m = groups.size();
	int parts = (n + 32 * 1024 - 1) / (32 * 1024);
	int range = parts * 1024;
	int blocks = (n + range - 1)/range;
	int *d_values, *d_groups, *d_groups_original;

	cudaMalloc(&d_values, n * sizeof(int));
	cudaMemcpyAsync(d_values, values.data(), n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&d_groups_original, m * sizeof(int));
	cudaMemcpyAsync(d_groups_original, groups.data(), m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&d_groups, n * sizeof(int));
	cudaMemset(d_groups, 0, n * sizeof(int));

	fill_groups<<<(m + 1023)/1024, 1024>>>(d_groups, d_groups_original, m);
	block_prefix_scan<<<blocks, 1024>>>(d_values, d_groups, n, parts, range);

	if (blocks > 1) {
		partials<<<1, 32>>>();
		propagate<<<blocks - 1, 1024>>>(d_values, d_groups, n, parts, range);
	}

	cudaMemcpy(values.data(), d_values, n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_values);
	cudaFree(d_groups);
	cudaFree(d_groups_original);
}
