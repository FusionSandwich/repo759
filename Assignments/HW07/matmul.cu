#include "matmul.cuh"
#include <stdio.h>

// Kernel for integer matrix multiplication
template <unsigned int TILE_DIM>
__global__ void matmul_kernel(const int *A, const int *B, int *C, unsigned int n) {
    // Shared memory for the tiles
    __shared__ int A_tile[TILE_DIM][TILE_DIM];
    __shared__ int B_tile[TILE_DIM][TILE_DIM];
    
    // Calculate row and column indices
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Accumulate result for C[row][col]
    int sum = 0;
    
    // Loop over tiles
    for (unsigned int t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tiles into shared memory
        if (row < n && t * TILE_DIM + threadIdx.x < n)
            A_tile[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_DIM + threadIdx.x];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0;
            
        if (t * TILE_DIM + threadIdx.y < n && col < n)
            B_tile[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * n + col];
        else
            B_tile[threadIdx.y][threadIdx.x] = 0;
            
        // Synchronize to ensure all threads have loaded the tiles
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (unsigned int k = 0; k < TILE_DIM; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        
        // Synchronize before loading next tiles
        __syncthreads();
    }
    
    // Write result to C
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Kernel for float matrix multiplication
template <unsigned int TILE_DIM>
__global__ void matmul_kernel(const float *A, const float *B, float *C, unsigned int n) {
    // Shared memory for the tiles
    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM];
    
    // Calculate row and column indices
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Accumulate result for C[row][col]
    float sum = 0.0f;
    
    // Loop over tiles
    for (unsigned int t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tiles into shared memory
        if (row < n && t * TILE_DIM + threadIdx.x < n)
            A_tile[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_DIM + threadIdx.x];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (t * TILE_DIM + threadIdx.y < n && col < n)
            B_tile[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * n + col];
        else
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
            
        // Synchronize to ensure all threads have loaded the tiles
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (unsigned int k = 0; k < TILE_DIM; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        
        // Synchronize before loading next tiles
        __syncthreads();
    }
    
    // Write result to C
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Kernel for double matrix multiplication
template <unsigned int TILE_DIM>
__global__ void matmul_kernel(const double *A, const double *B, double *C, unsigned int n) {
    // Shared memory for the tiles
    __shared__ double A_tile[TILE_DIM][TILE_DIM];
    __shared__ double B_tile[TILE_DIM][TILE_DIM];
    
    // Calculate row and column indices
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Accumulate result for C[row][col]
    double sum = 0.0;
    
    // Loop over tiles
    for (unsigned int t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tiles into shared memory
        if (row < n && t * TILE_DIM + threadIdx.x < n)
            A_tile[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_DIM + threadIdx.x];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0;
            
        if (t * TILE_DIM + threadIdx.y < n && col < n)
            B_tile[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * n + col];
        else
            B_tile[threadIdx.y][threadIdx.x] = 0.0;
            
        // Synchronize to ensure all threads have loaded the tiles
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (unsigned int k = 0; k < TILE_DIM; ++k) {
            sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
        }
        
        // Synchronize before loading next tiles
        __syncthreads();
    }
    
    // Write result to C
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Host function for integer matrix multiplication
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n, unsigned int block_dim) {
    // Configure grid and block dimensions
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    
    // Call the kernel based on block dimension
    if (block_dim == 32) {
        matmul_kernel<32><<<dimGrid, dimBlock>>>(A, B, C, n);
    } else if (block_dim == 16) {
        matmul_kernel<16><<<dimGrid, dimBlock>>>(A, B, C, n);
    } else if (block_dim == 8) {
        matmul_kernel<8><<<dimGrid, dimBlock>>>(A, B, C, n);
    } else {
        // For other block dimensions (this can be slow due to dynamic allocation)
        printf("Warning: Non-optimized block dimension %u. Using default tile size.\n", block_dim);
        
        // Call a specialized kernel with default tile size
        matmul_kernel<32><<<dimGrid, dimBlock>>>(A, B, C, n);
    }
    
    // Synchronize for timing purposes
    cudaDeviceSynchronize();
}

// Host function for float matrix multiplication
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n, unsigned int block_dim) {
    // Configure grid and block dimensions
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    
    // Call the kernel based on block dimension
    if (block_dim == 32) {
        matmul_kernel<32><<<dimGrid, dimBlock>>>(A, B, C, n);
    } else if (block_dim == 16) {
        matmul_kernel<16><<<dimGrid, dimBlock>>>(A, B, C, n);
    } else if (block_dim == 8) {
        matmul_kernel<8><<<dimGrid, dimBlock>>>(A, B, C, n);
    } else {
        // For other block dimensions (this can be slow due to dynamic allocation)
        printf("Warning: Non-optimized block dimension %u. Using default tile size.\n", block_dim);
        
        // Call a specialized kernel with default tile size
        matmul_kernel<32><<<dimGrid, dimBlock>>>(A, B, C, n);
    }
    
    // Synchronize for timing purposes
    cudaDeviceSynchronize();
}

// Host function for double matrix multiplication
__host__ void matmul_3(const double *A, const double *B, double *C, unsigned int n, unsigned int block_dim) {
    // Configure grid and block dimensions
    dim3 dimBlock(block_dim, block_dim);
    dim3 dimGrid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim);
    
    // Call the kernel based on block dimension
    if (block_dim == 32) {
        matmul_kernel<32><<<dimGrid, dimBlock>>>(A, B, C, n);
    } else if (block_dim == 16) {
        matmul_kernel<16><<<dimGrid, dimBlock>>>(A, B, C, n);
    } else if (block_dim == 8) {
        matmul_kernel<8><<<dimGrid, dimBlock>>>(A, B, C, n);
    } else {
        // For other block dimensions (this can be slow due to dynamic allocation)
        printf("Warning: Non-optimized block dimension %u. Using dynamic shared memory.\n", block_dim);
        
        // Call a specialized kernel (same compromise as above)
        matmul_kernel<32><<<dimGrid, dimBlock>>>(A, B, C, n);
    }
    
    // Synchronize for timing purposes
    cudaDeviceSynchronize();
}