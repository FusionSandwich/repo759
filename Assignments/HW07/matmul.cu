// File: matmul.cu
// Implements tiled matrix multiplication using shared memory.

#include "matmul.cuh"
#include <cuda_runtime.h>
#include <cmath>    // For ceil
#include <cstdio>   // For fprintf, stderr

// CUDA Kernel for Tiled Matrix Multiplication
// Template allows use with int, float, double
template <typename T>
__global__ void matmul_kernel(const T *A, const T *B, T *C,
                              unsigned int n,
                              unsigned int block_dim) {
    // Calculate the row and column index of the element handled by this thread
    unsigned int row = blockIdx.y * block_dim + threadIdx.y;
    unsigned int col = blockIdx.x * block_dim + threadIdx.x;

    // Shared memory tiles for sub-matrices of A and B
    __shared__ T As[32][32];  // Assuming block_dim <= 32
    __shared__ T Bs[32][32];

    T Cvalue = 0;  // Accumulator for the element C(row, col)
    unsigned int num_tiles = (n + block_dim - 1) / block_dim;

    // Loop over the tiles of A and B required to compute C(row, col)
    for (unsigned int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        unsigned int A_row_idx = blockIdx.y * block_dim + threadIdx.y;
        unsigned int A_col_idx = tile_idx * block_dim + threadIdx.x;
        unsigned int B_row_idx = tile_idx * block_dim + threadIdx.y;
        unsigned int B_col_idx = blockIdx.x * block_dim + threadIdx.x;

        // Load tile of A (with bounds check)
        if (A_row_idx < n && A_col_idx < n) {
            As[threadIdx.y][threadIdx.x] = A[A_row_idx * n + A_col_idx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }

        // Load tile of B (with bounds check)
        if (B_row_idx < n && B_col_idx < n) {
            Bs[threadIdx.y][threadIdx.x] = B[B_row_idx * n + B_col_idx];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Multiply the two tiles
        for (unsigned int k = 0; k < block_dim; ++k) {
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write back the result
    if (row < n && col < n) {
        C[row * n + col] = Cvalue;
    }
}

// Host function template to launch the kernel
template <typename T>
__host__ void matmul_host(const T *A, const T *B, T *C,
                          unsigned int n,
                          unsigned int block_dim) {
    dim3 blockDim(block_dim, block_dim);
    unsigned int grid_size = (n + block_dim - 1) / block_dim;
    dim3 gridDim(grid_size, grid_size);

    matmul_kernel<T><<<gridDim, blockDim>>>(A, B, C, n, block_dim);
    cudaDeviceSynchronize();

    // Check for errors after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "CUDA error after kernel launch: %s\n",
                cudaGetErrorString(err));
    }
}

// Explicit instantiations for int, float, and double
__host__ void matmul_1(const int *A, const int *B, int *C,
                       unsigned int n, unsigned int block_dim) {
    matmul_host<int>(A, B, C, n, block_dim);
}

__host__ void matmul_2(const float *A, const float *B, float *C,
                       unsigned int n, unsigned int block_dim) {
    matmul_host<float>(A, B, C, n, block_dim);
}

__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim) {
    matmul_host<double>(A, B, C, n, block_dim);
}
