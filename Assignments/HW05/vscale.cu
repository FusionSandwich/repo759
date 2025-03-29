#include "vscale.cuh"

__global__ void vscale(const float *a, float *b, unsigned int n) {
    // Compute global thread ID
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread handles at most one element
    if (idx < n) {
        b[idx] = a[idx] * b[idx];
    }
}
