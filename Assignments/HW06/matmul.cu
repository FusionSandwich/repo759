// matmul.cu
#include "matmul.cuh"

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure thread index is within matrix bounds
    if (idx < n * n) {
        // Map 1D index to 2D row and column
        int row = idx / n;
        int col = idx % n;
        
        // Compute dot product for C[row][col]
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block) {
    // Calculate number of blocks to cover all n*n elements
    int num_blocks = (n * n + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel with 1D grid
    matmul_kernel<<<num_blocks, threads_per_block>>>(A, B, C, n);
}