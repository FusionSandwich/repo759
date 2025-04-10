#include <cuda_runtime.h>
#include "matmul.cuh"

// Computes the matrix product of A and B, storing the result in C.
// Each thread should compute _one_ element of output.
// Does not use shared memory for this problem.
//
// A, B, and C are row major representations of nxn matrices in device memory.
//
// Assumptions:
// - 1D kernel configuration
__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    // Calculate the row and column index for the element this thread will compute
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Check bounds to ensure we don't go outside the matrix dimensions
    // Since it's a 1D kernel, we need to map the 1D index to 2D C matrix indices
    // Let's assume the 1D grid covers all elements of C linearly (row-major)
    int total_elements = n * n;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < total_elements) {
        int r = tid / n; // Row index in C
        int c = tid % n; // Column index in C
        float sum = 0.0f;

        // Compute dot product for C[r][c]
        for (size_t k = 0; k < n; ++k) {
            // A[r][k] is at index r*n + k
            // B[k][c] is at index k*n + c
            sum += A[r * n + k] * B[k * n + c];
        }
        C[tid] = sum; // Store result in C at the linear index tid
    }
}


// Makes one call to matmul_kernel with threads_per_block threads per block.
// You can consider following the kernel call with cudaDeviceSynchronize (but if you use
// cudaEventSynchronize to time it, that call serves the same purpose as cudaDeviceSynchronize).
void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    size_t total_elements = n * n;
    // Calculate the number of blocks needed
    unsigned int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // Launch the kernel
    matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);

    // Optional: Synchronize device if not using events for timing elsewhere
    // cudaDeviceSynchronize();
}