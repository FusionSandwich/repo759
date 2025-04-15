#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

// Debug kernel using global memory to isolate shared memory issues
__global__ void stencil_kernel(const float* image, float* output, const float* mask, 
                              unsigned int n, unsigned int R) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float sum = 0.0f;
        for (int j = -R; j <= R; j++) {
            int idx = i + j;
            float img_val = (idx >= 0 && idx < n) ? image[idx] : 1.0f;
            sum += img_val * mask[j + R];
        }
        output[i] = sum;
    }
}

int main() {
    unsigned int n = 5;
    unsigned int R = 1;
    unsigned int threads_per_block = 3;
    unsigned int blocks = (n + threads_per_block - 1) / threads_per_block;

    // Host arrays
    float h_image[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float h_mask[] = {0.1f, 0.2f, 0.3f};
    float* h_output = new float[n];

    // Device arrays
    float *d_image, *d_output, *d_mask;
    CHECK_CUDA(cudaMalloc(&d_image, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mask, (2 * R + 1) * sizeof(float)));

    // Initialize d_output to -1.0 to detect if kernel writes
    CHECK_CUDA(cudaMemset(d_output, 0xBF, n * sizeof(float))); // 0xBF sets float to -1.0

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_image, h_image, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    stencil_kernel<<<blocks, threads_per_block>>>(d_image, d_output, d_mask, n, R);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print output
    std::cout << "Output: ";
    for (unsigned int i = 0; i < n; ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CHECK_CUDA(cudaFree(d_image));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_mask));
    delete[] h_output;

    return 0;
}