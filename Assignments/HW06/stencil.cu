#include <cuda_runtime.h>
#include "stencil.cuh"

// Kernel: Each thread computes one element of the output array.
__global__ void stencil_kernel(const float* image, const float* mask, float* output, 
                               unsigned int n, unsigned int R) {
    // Allocate dynamic shared memory.
    // The first portion holds the entire mask (size: 2*R+1).
    // The remainder holds the image tile needed for this block (size: blockDim.x + 2*R).
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + tid;
    
    // Pointers into shared memory.
    float* sh_mask = shared; 
    float* sh_tile = (float*)(shared + (2 * R + 1));

    // Load the entire mask into shared memory.
    if (tid < (2 * R + 1)) {
        sh_mask[tid] = mask[tid];
    }
    __syncthreads();

    // Compute starting index for the image tile.
    int tile_start = blockIdx.x * blockDim.x - R;
    // Total number of image elements needed by the block.
    int tile_size = blockDim.x + 2 * R;
    // Each thread cooperatively loads the tile.
    for (int i = tid; i < tile_size; i += blockDim.x) {
        int img_index = tile_start + i;
        // Use image value of 1.0 if out-of-bound.
        if (img_index < 0 || img_index >= n)
            sh_tile[i] = 1.0f;
        else
            sh_tile[i] = image[img_index];
    }
    __syncthreads();

    // Perform convolution if within valid global index.
    if (global_index < n) {
        float sum = 0.0f;
        // The corresponding index in the shared tile.
        int tile_idx = tid + R;
        // Compute the sum for indices j from -R to R.
        for (int j = -int(R); j <= int(R); j++) {
            sum += sh_tile[tile_idx + j] * sh_mask[j + R];
        }
        output[global_index] = sum;
    }
}

// Host function that launches the kernel.
void stencil(const float* image, const float* mask, float* output, 
             unsigned int n, unsigned int R, unsigned int threads_per_block) {
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    // Allocate shared memory: space for mask and image tile.
    size_t shared_size = ((2 * R + 1) + (threads_per_block + 2 * R)) * sizeof(float);
    stencil_kernel<<<blocks, threads_per_block, shared_size>>>(image, mask, output, n, R);
}
