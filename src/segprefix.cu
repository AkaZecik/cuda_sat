#include <cstdio>
#include "segprefix.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

// lX - literal X
// vX - value of literal X
// sX - sign of literal X
#define l1(n) (((n) & 0xff000000u) >> 24)
#define l2(n) (((n) & 0x00ff0000u) >> 16)
#define l3(n) (((n) & 0x0000ff00u) >> 8)
#define v1(n) (((n) & 0x00000080u) >> 7)
#define v2(n) (((n) & 0x00000040u) >> 6)
#define v3(n) (((n) & 0x00000020u) >> 5)
#define s1(n) (((n) & 0x00000010u) >> 4)
#define s2(n) (((n) & 0x00000008u) >> 3)
#define s3(n) (((n) & 0x00000004u) >> 2)

#define unset_l1(n) ((n) & 0x00ffffffu)
#define unset_l2(n) ((n) & 0xff00ffffu)
#define unset_l3(n) ((n) & 0xffff00ffu)
#define unset_v1(n) ((n) & 0xffffff7fu)
#define unset_v2(n) ((n) & 0xffffffbfu)
#define unset_v3(n) ((n) & 0xffffffdfu)
#define unset_s1(n) ((n) & 0xffffffefu)
#define unset_s2(n) ((n) & 0xfffffff7u)
#define unset_s3(n) ((n) & 0xfffffffbu)

#define set_l1(n, v) (unset_l1(n) | (((v) & 0x000000ffu) << 24))
#define set_l2(n, v) (unset_l2(n) | (((v) & 0x000000ffu) << 16))
#define set_l3(n, v) (unset_l3(n) | (((v) & 0x000000ffu) << 8))
#define set_v1(n) ((n) | 0x00000080u)
#define set_v2(n) ((n) | 0x00000040u)
#define set_v3(n) ((n) | 0x00000020u)
#define set_s1(n) ((n) | 0x00000010u)
#define set_s2(n) ((n) | 0x00000008u)
#define set_s3(n) ((n) | 0x00000004u)

typedef unsigned int uint;
typedef unsigned char uchar;

__global__ void sat_kernel(uint *d_arr, uint *d_valid, int s, int k, int n, int r) {
	int warps_nb = (blockDim.x + 31) >> 5;
	int warp_id = warps_nb * blockIdx.x + (threadIdx.x >> 5);
	int formula_id = warp_id / 3;
	int copy_id = warp_id - 3 * formula_id;
	uint *formula = d_arr + formula_id * (n + r);
	uint *assignments = formula + r;
	uint first_clause = formula[0];
	int lit1 = l1(first_clause);
	int lit2 = l2(first_clause);
	int lit3 = l3(first_clause);

	if(copy_id == 0) {
		if(a == 0) {
			return;
		}

		for(int i = threadIdx.x & 31; i < r; i += 32) {
			uint clause = formula[i];
		}
	}

	if(copy_id == 1) {
		if(b == 0) {
			return;
		}

		for(int i = threadIdx.x & 31; i < r; i += 32) {
			uint clause = formula[i];
		}
	}

	if(copy_id == 2) {
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
