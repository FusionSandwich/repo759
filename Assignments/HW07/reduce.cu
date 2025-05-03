#include "reduce.cuh"
#include <stdio.h> // Add this include for printf

/**
 * Kernel for parallel reduction with first add during global load optimization (Kernel 4)
 * @param g_idata - Input array on device
 * @param g_odata - Output array on device
 * @param n - Number of elements in the input array
 */
__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    // Allocate shared memory dynamically
    extern __shared__ float sdata[];
    
    // Thread and block index
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Clear shared memory location
    sdata[tid] = 0;
    
    // Load and add first element if in bounds
    if (i < n) {
        sdata[tid] = g_idata[i];
    }
    
    // Load and add second element if in bounds (first add during load)
    if (i + blockDim.x < n) {
        sdata[tid] += g_idata[i + blockDim.x];
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

/**
 * Host function to perform complete reduction
 * @param input - Input array on device memory
 * @param output - Output array on device memory
 * @param N - Number of elements in the input array
 * @param threads_per_block - Number of threads per block
 */
__host__ void reduce(float **input, float **output, unsigned int N, unsigned int threads_per_block) {
    // Input and output arrays for the current reduction step
    float *current_input = *input;
    float *current_output = *output;
    
    // Size of the current input array
    unsigned int current_size = N;
    
    // Keep reducing until we have just one element
    while (current_size > 1) {
        // Number of blocks needed for the current kernel call
        // Each thread processes 2 elements, so we need (current_size + 2*threads_per_block - 1)/(2*threads_per_block) blocks
        unsigned int num_blocks = (current_size + 2 * threads_per_block - 1) / (2 * threads_per_block);
        
        // If the number of blocks is too large, limit it to avoid kernel launch failures
        const unsigned int MAX_BLOCKS = 65535; // Maximum number of blocks in a grid dimension
        if (num_blocks > MAX_BLOCKS) {
            num_blocks = MAX_BLOCKS;
        }
        
        // Launch the kernel with dynamically allocated shared memory
        reduce_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
            current_input, current_output, current_size);
        
        // Check for kernel launch errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            break;
        }
        
        // Update size for the next reduction
        current_size = num_blocks;
        
        // Swap input and output pointers for the next iteration
        float *temp = current_input;
        current_input = current_output;
        current_output = temp;
    }
    
    // If the final result is in the output buffer, copy it to the first element of the input
    if (current_input != *input) {
        cudaMemcpy(*input, current_input, sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // For timing purposes
    cudaDeviceSynchronize();
}