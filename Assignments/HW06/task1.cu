// task1.cu
#include <iostream>
#include <cuda_runtime.h>
#include "matmul.cuh"
#include <random>
#include <iomanip>

int main(int argc, char** argv) {
    // Check command-line arguments
    if (argc != 3) {
        std::cerr << "Usage: ./task1 n threads_per_block" << std::endl;
        return 1;
    }

    size_t n = std::stoi(argv[1]);
    unsigned int threads_per_block = std::stoi(argv[2]);

    // Allocate host memory
    float* h_A = new float[n * n];
    float* h_B = new float[n * n];
    float* h_C = new float[n * n];

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    // Fill host matrices with random numbers
    for (size_t i = 0; i < n * n; i++) {
        h_A[i] = dis(gen);
        h_B[i] = dis(gen);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * n * sizeof(float));
    cudaMalloc(&d_B, n * n * sizeof(float));
    cudaMalloc(&d_C, n * n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    // Launch matrix multiplication
    matmul(d_A, d_B, d_C, n, threads_per_block);

    // Record stop time
    cudaEventRecord(stop);

    // Wait for kernel to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print last element and time with 2 decimal places
    std::cout << std::fixed << std::setprecision(2) << h_C[n * n - 1] << std::endl;
    std::cout << std::fixed << std::setprecision(2) << milliseconds << std::endl;

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}