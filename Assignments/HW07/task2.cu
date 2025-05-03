#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include "reduce.cuh"

// Function to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " N threads_per_block" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    unsigned long long N_ull = std::stoull(argv[1]);
    unsigned int N = (unsigned int)N_ull;  // Convert to unsigned int safely
    unsigned int threads_per_block = std::stoi(argv[2]);
    
    // Ensure threads_per_block is valid
    if (threads_per_block > 1024) {
        std::cerr << "Error: threads_per_block cannot exceed 1024" << std::endl;
        return 1;
    }
    
    // Create and fill array on host with random numbers in [-1, 1]
    float *h_input = new float[N];
    
    // Set up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // Fill array with random values
    for (unsigned int i = 0; i < N; ++i) {
        h_input[i] = dist(gen);
    }
    
    // Calculate CPU sum for verification (optional)
    float cpu_sum = 0.0f;
    for (unsigned int i = 0; i < N; ++i) {
        cpu_sum += h_input[i];
    }
    
    // Allocate memory on device for input array
    float *d_input;
    cudaError_t err = cudaMalloc(&d_input, N * sizeof(float));
    checkCudaError(err, "Failed to allocate device memory for input array");
    
    // Copy input data from host to device
    err = cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy input data from host to device");
    
    // Calculate number of blocks needed for the first call to reduce_kernel
    unsigned int num_blocks = (N + 2 * threads_per_block - 1) / (2 * threads_per_block);
    
    // Limit number of blocks to avoid kernel launch failures
    const unsigned int MAX_BLOCKS = 65535;
    if (num_blocks > MAX_BLOCKS) {
        num_blocks = MAX_BLOCKS;
    }
    
    // Allocate memory on device for output array
    float *d_output;
    err = cudaMalloc(&d_output, num_blocks * sizeof(float));
    checkCudaError(err, "Failed to allocate device memory for output array");
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timing
    cudaEventRecord(start);
    
    // Call reduce function
    reduce(&d_input, &d_output, N, threads_per_block);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    float result;
    err = cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy result from device to host");
    
    // Print result and time
    // Make sure we flush after each output to ensure it's written correctly
    std::cout << result << std::endl << std::flush;
    std::cout << milliseconds << std::endl << std::flush;
    
    // Cleanup
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}