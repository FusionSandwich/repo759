#include "matmul.cuh"
#include <cuda_runtime.h>
#include <stdio.h> 
// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Renamed from 'row' to 'idx'

    // Check bounds to ensure we don't go past the matrix dimensions
    // This is necessary for a 1D grid launch when n*n is not a multiple of blockDim.x
    if (idx < n * n) { // Use 'idx' for the check
        int r = idx / n; // Calculate output row index using 'idx'
        int c = idx % n; // Calculate output column index using 'idx'
        float sum = 0.0f;

        // Compute the dot product for element C[r][c]
        for (size_t k = 0; k < n; ++k) {
            sum += A[r * n + k] * B[k * n + c];
        }
        C[r * n + c] = sum;
    }
}

// Makes one call to matmul_kernel with threads_per_block threads per block.
void matmul(const float* d_A, const float* d_B, float* d_C, size_t n, unsigned int threads_per_block) {
    // Calculate the total number of elements (threads needed)
    size_t total_elements = n * n;

    // Calculate the number of blocks needed
    // Ceiling division: (total_elements + threads_per_block - 1) / threads_per_block
    unsigned int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    matmul_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, n);

    // It's good practice to check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        // Consider exiting or handling the error appropriately
    }

}
