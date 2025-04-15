#include <iostream>
#include <cstdlib>
#include <ctime>          // Include to use time() for seeding
#include <cuda_runtime.h>
#include "stencil.cuh"

int main(int argc, char* argv[]) {
    // Expect exactly three arguments: n, R, and threads_per_block.
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " n R threads_per_block" << std::endl;
        return 1;
    }
    unsigned int n = std::atoi(argv[1]);
    unsigned int R = std::atoi(argv[2]);
    unsigned int threads_per_block = std::atoi(argv[3]);

    // Seed the random number generator with the current time so that we get different numbers each run.
    srand(static_cast<unsigned int>(time(NULL)));

    // Allocate host arrays.
    float* h_image = new float[n];
    float* h_output = new float[n];
    float* h_mask = new float[2 * R + 1];

    // Initialize the image array with random numbers in the range [-1, 1].
    for (unsigned int i = 0; i < n; ++i) {
        h_image[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    }
    // Initialize the mask array with random numbers in the range [-1, 1].
    for (unsigned int i = 0; i < 2 * R + 1; ++i) {
        h_mask[i] = 2.0f * rand() / RAND_MAX - 1.0f;
    }

    // Allocate device memory.
    float *d_image, *d_mask, *d_output;
    cudaMalloc(&d_image, n * sizeof(float));
    cudaMalloc(&d_mask, (2 * R + 1) * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    // Copy data from host to device.
    cudaMemcpy(d_image, h_image, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, (2 * R + 1) * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing the stencil execution.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Call the stencil function.
    stencil(d_image, d_mask, d_output, n, R, threads_per_block);

    // Record the end event.
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Copy the result back to host.
    cudaMemcpy(h_output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the last element of the output array.
    std::cout << h_output[n - 1] << std::endl;
    // Print the kernel execution time (in milliseconds).
    std::cout << elapsed_time << std::endl;

    // Free allocated memory.
    delete[] h_image;
    delete[] h_mask;
    delete[] h_output;
    cudaFree(d_image);
    cudaFree(d_mask);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
