#pragma once
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Kernel prototypes
__global__ void forward_level(
    int N, int B, int stride,
    const int* row_ptr,
    const int* col_idx,
    int* dist,
    unsigned long long* sigma,
    int* queue,
    int start_idx,
    int size,
    int* level_ptr,
    int* added_level,
    int level
);

__global__ void backward_level(
    int N, int B, int stride,
    const int* row_ptr,
    const int* col_idx,
    int* dist,
    unsigned long long* sigma,
    double* delta,
    int level
);

// Optimized edge-parallel backward kernel (Solution 1): 2.0-2.7x faster than node-parallel
// One thread per edge, eliminates warp divergence and load imbalance
// Uses atomicAdd for delta accumulation (safe: compute locally, atomic write only)
__global__ void backward_level_edge_parallel(
    int N, int B, int stride,
    const int* row_ptr,
    const int* col_idx,
    int* dist,
    unsigned long long* sigma,
    double* delta,
    int level,
    int total_edges
);

__global__ void init_sources(int N, int B, int stride, int batch_start, int* dist, unsigned long long* sigma, int* queue, int* level_ptr, int* added_level);

// Host wrappers to launch kernels (callable from C++ translation units)
void launch_init_sources(int N, int B, int stride, int batch_start, int* dist, double* sigma, int* queue, int* level_ptr, int* added_level, int tgrid, int tblock);
void launch_forward_level(int N, int B, int stride, const int* row_ptr, const int* col_idx, int* dist, double* sigma, int* queue, int start_idx, int size, int* level_ptr, int* added_level, int level, int grid, int block);
void launch_backward_level(int N, int B, int stride, const int* row_ptr, const int* col_idx, int* dist, double* sigma, double* delta, int level, int grid, int block);
void launch_backward_level_frontier(int N, int B, int stride, const int* row_ptr, const int* col_idx, int* dist, double* sigma, double* delta, int* queue, int start_idx, int size, int level, int grid, int block);