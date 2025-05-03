// File: reduce.cu
#include "reduce.cuh"
#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// Kernel 4: First Add During Load
// Each thread loads two elements, sums them into shared memory, then does
// a tree‐based in‐block reduction.
// ----------------------------------------------------------------------------
__global__
void reduce_kernel(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + tid;

    // First add during global load:
    float sum = 0.0f;
    if (idx < n) {
        sum = g_idata[idx];
        if (idx + blockDim.x < n) {
            sum += g_idata[idx + blockDim.x];
        }
    }
    sdata[tid] = sum;
    __syncthreads();

    // In‐block tree reduction:
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block’s result
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}

// ----------------------------------------------------------------------------
// Host‐side reduction driver
//   *input, *output: addresses of device buffers
//   N: initial length of *input
//   threads_per_block: number of threads per block to launch
//
// Repeatedly invokes reduce_kernel until we have a single sum.
// At the end *input is updated to point at the buffer whose [0] holds the total.
// Ends with cudaDeviceSynchronize() for accurate timing.
// ----------------------------------------------------------------------------
__host__
void reduce(float **input, float **output,
            unsigned int N, unsigned int threads_per_block)
{
    unsigned int num_elements = N;
    float *in_ptr  = *input;
    float *out_ptr = *output;

    // How many blocks for the first pass?
    unsigned int blocks = (num_elements + threads_per_block * 2 - 1)
                              / (threads_per_block * 2);

    // Shared memory size per block
    size_t shared_mem = threads_per_block * sizeof(float);

    // Keep reducing until we get down to 1 block
    while (blocks > 1) {
        reduce_kernel<<<blocks, threads_per_block, shared_mem>>>(
            in_ptr, out_ptr, num_elements);
        cudaDeviceSynchronize();

        num_elements = blocks;
        blocks = (num_elements + threads_per_block * 2 - 1)
                     / (threads_per_block * 2);

        // swap input and output pointers
        float *tmp = in_ptr;
        in_ptr  = out_ptr;
        out_ptr = tmp;
    }

    // Final pass: blocks == 1
    reduce_kernel<<<blocks, threads_per_block, shared_mem>>>(
        in_ptr, out_ptr, num_elements);
    cudaDeviceSynchronize();

    // Now out_ptr[0] holds the total sum; make *input point to it
    *input = out_ptr;
}
