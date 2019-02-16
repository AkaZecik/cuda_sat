#include <cstdio>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define WARPS_NB 10

typedef unsigned int uint;
typedef unsigned char uchar;

struct clause {
	uchar l1;
	uchar l2;
	uchar l3;
	bool s1:1;
	bool s2:1;
	bool s3:1;
	bool v1:1;
	bool v2:1;
	bool v3:1;
};

/* Triples the number of formulas in a batch and marks invalid/missing ones
 * 
 * d_f - array of formulas and a triple of a free space for new formulas
 * d_v - array of flags indicating whether a formula is valid or not
 * k - number of formulas to triple
 * n - total number of literals
 * r - total number of clauses
 */
__global__ void sat_kernel(clause *d_f, uint *d_v, int k, int n, int r) {
	int warp_id = WARPS_NB * blockIdx.x + (threadIdx.x >> 5);
	int formula_id = warp_id / 3;
	int branch_id = warp_id - 3 * formula_id;
	uint *formula = d_f + formula_id * r;
	clause fc = formula[0]; // first clause

	if(branch_id == 0) {
		if(fc.s1) {
			return;
		}

		for(int i = threadIdx.x & 31; i < r; i += 32) {
			clause c = formula[i];

			if(!c.s1) {
				if(c.l1 == fc.l1) {
					c.v1 = fc.v1;
				}

				if(c.l1 == fc.l2) {
					c.v1 = fc.v2;
				}

				if(c.l1 == fc.l3) {
					c.v1 = fc.v3;
				}
			}
		}
	}

	if(branch_id == 1) {
		if(b == 0) {
			return;
		}

		for(int i = threadIdx.x & 31; i < r; i += 32) {
			uint clause = formula[i];
		}
	}

	if(branch_id == 2) {
		if(c == 0) {
			return;
		}

		for(int i = threadIdx.x & 31; i < r; i += 32) {
			uint clause = formula[i];
		}
	}
}

__global__ void 1d_scan() {

}

__global__ void 2d_scan() {

}

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
