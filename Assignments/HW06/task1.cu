// Based on instructions in HW06.pdf [cite: 12, 13, 14, 15]
#include <iostream>
#include <vector>
#include <cstdlib> // For atoi, srand, rand
#include <ctime>   // For time
#include <cuda_runtime.h>
#include "matmul.cuh" // Include the header file

// Helper function to check CUDA calls
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n threads_per_block" << std::endl;
        return 1;
    }

    size_t n = std::atoi(argv[1]);
    unsigned int threads_per_block = std::atoi(argv[2]);

    if (n <= 0 || threads_per_block <= 0) {
        std::cerr << "n and threads_per_block must be positive integers." << std::endl;
        return 1;
    }

    size_t matrix_size_bytes = n * n * sizeof(float);

    // Allocate host memory
    std::vector<float> h_A(n * n);
    std::vector<float> h_B(n * n);
    std::vector<float> h_C(n * n);

    // Initialize host matrices with random numbers [-1, 1] [cite: 12]
    srand(time(0)); // Seed random number generator
    for (size_t i = 0; i < n * n; ++i) {
        h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Allocate device memory [cite: 13]
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, matrix_size_bytes));
    CHECK_CUDA(cudaMalloc(&d_B, matrix_size_bytes));
    CHECK_CUDA(cudaMalloc(&d_C, matrix_size_bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), matrix_size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), matrix_size_bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing [cite: 15]
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start));

    // Call the matmul function [cite: 14]
    matmul(d_A, d_B, d_C, n, threads_per_block);

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop));

    // Synchronize to wait for the event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result from device to host
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, matrix_size_bytes, cudaMemcpyDeviceToHost));

    // Print the last element of the resulting matrix [cite: 14]
    std::cout << h_C[n * n - 1] << std::endl;

    // Print the time taken [cite: 15]
    std::cout << milliseconds << std::endl;

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}