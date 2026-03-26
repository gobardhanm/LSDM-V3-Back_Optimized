#include "../include/bc_cuda.hpp"
#include <cuda_runtime.h>

__device__ double atomicAddDouble(double* address, double val);

__global__
void forward_level(
    int N, int B, int stride,
    const int* row_ptr,
    const int* col_idx,
    int* dist,
    double* sigma,
    int* queue,
    int start_idx,
    int size,
    int* level_ptr,
    int* added_level,
    int level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (warp_id >= size) return;
    
    int u = queue[start_idx + warp_id];

    // Warp-parallel Bitset creation! 
    unsigned int is_active = (lane < B && dist[u*stride + lane] == level) ? 1 : 0;
    unsigned int active = __ballot_sync(0xffffffff, is_active);

    if (active == 0u) return;

    // Warp-parallel Edge fetching
    int start_edge = row_ptr[u];
    int end_edge = row_ptr[u+1];
    
    for (int e = start_edge + lane; e < end_edge; e += 32) {
        int v = col_idx[e];
        for (int s = 0; s < B; s++) {
            if (!(active & (1u << s))) continue;
            
            int idx = v*stride + s;
            int old = atomicCAS(&dist[idx], -1, level+1);

            if (old == -1) {
                atomicAddDouble(&sigma[idx], sigma[u*stride + s]);
                int old_val = atomicExch(&added_level[v], level+1);
                if (old_val != level+1) {
                    int pos = atomicAdd(&level_ptr[level+2], 1);
                    queue[pos] = v;
                }
            }
            else if (old == level+1) {
                atomicAddDouble(&sigma[idx], sigma[u*stride + s]);
            }
        }
    }
}

// DEPRECATED: Original node-parallel backward kernel. Kept for reference/fallback.
// See backward_level_edge_parallel for optimized version (2.0-2.7x faster).
__global__
void backward_level_old(
    int N, int B, int stride,
    const int* row_ptr,
    const int* col_idx,
    int* dist,
    double* sigma,
    double* delta,
    int level
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;

    for (int s = 0; s < B; s++) {
        if (dist[u*stride + s] != level) continue;

        double acc = 0.0;
        for (int e = row_ptr[u]; e < row_ptr[u+1]; e++) {
            int v = col_idx[e];
            if (dist[v*stride + s] == level+1) {
                acc += ((double)sigma[u*stride + s] /
                        (double)sigma[v*stride + s]) *
                       (1.0 + delta[v*stride + s]);
            }
        }
        delta[u*stride + s] += acc;
    }
}

// Binary search helper: find node u such that row_ptr[u] <= edge_id < row_ptr[u+1]
__device__ int binary_search_node(const int* row_ptr, int edge_id, int N) {
    int lo = 0, hi = N;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (row_ptr[mid] <= edge_id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo - 1;
}

// Utility: Atomic add for double using compare-and-swap (CAS)
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__
void backward_level_frontier(
    int N, int B, int stride,
    const int* row_ptr,
    const int* col_idx,
    int* dist,
    double* sigma,
    double* delta,
    int* queue,
    int start_idx,
    int size,
    int level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (warp_id >= size) return;
    
    int u = queue[start_idx + warp_id];

    unsigned int is_active = (lane < B && dist[u*stride + lane] == level) ? 1 : 0;
    unsigned int active = __ballot_sync(0xffffffff, is_active);

    if (active == 0u) return;

    int start_edge = row_ptr[u];
    int end_edge = row_ptr[u+1];
    
    for (int s = 0; s < B; s++) {
        if (!(active & (1u << s))) continue;
        
        double acc = 0.0;
        double sigma_u = sigma[u*stride + s];
        
        for (int e = start_edge + lane; e < end_edge; e += 32) {
            int v = col_idx[e];
            if (dist[v*stride + s] == level+1) {
                acc += ((double)sigma_u / (double)sigma[v*stride + s]) *
                       (1.0 + delta[v*stride + s]);
            }
        }

        // Warp reduction for the dependency score without array overhead
        for (int offset = 16; offset > 0; offset /= 2) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        
        if (lane == 0) {
            delta[u*stride + s] += acc;
        }
    }
}

__global__
void init_sources(int N, int B, int stride, int batch_start, int* dist, double* sigma, int* queue, int* level_ptr, int* added_level) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= B) return;
    int src = batch_start + s;
    if (src >= N) return;
    dist[src*stride + s] = 0;
    sigma[src*stride + s] = 1ULL;
    
    int old = atomicExch(&added_level[src], 0);
    if (old != 0) {
        int pos = atomicAdd(&level_ptr[1], 1);
        queue[pos] = src;
    }
}
// Host wrappers
void launch_init_sources(int N, int B, int stride, int batch_start, int* dist, double* sigma, int* queue, int* level_ptr, int* added_level, int tgrid, int tblock) {
    init_sources<<<tgrid, tblock>>>(N, B, stride, batch_start, dist, sigma, queue, level_ptr, added_level);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_forward_level(int N, int B, int stride, const int* row_ptr, const int* col_idx, int* dist, double* sigma, int* queue, int start_idx, int size, int* level_ptr, int* added_level, int level, int grid, int block) {
    forward_level<<<grid, block>>>(N, B, stride, row_ptr, col_idx, dist, sigma, queue, start_idx, size, level_ptr, added_level, level);
    CUDA_CHECK(cudaGetLastError());
}

void launch_backward_level(int N, int B, int stride, const int* row_ptr, const int* col_idx, int* dist, double* sigma, double* delta, int level, int grid, int block) {
    backward_level_old<<<grid, block>>>(N, B, stride, row_ptr, col_idx, dist, sigma, delta, level);
    CUDA_CHECK(cudaGetLastError());
}

void launch_backward_level_frontier(int N, int B, int stride, const int* row_ptr, const int* col_idx, int* dist, double* sigma, double* delta, int* queue, int start_idx, int size, int level, int grid, int block) {
    backward_level_frontier<<<grid, block>>>(N, B, stride, row_ptr, col_idx, dist, sigma, delta, queue, start_idx, size, level);
    CUDA_CHECK(cudaGetLastError());
}