#include "matmul.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <cstdlib>

// Function to check CUDA errors
#define CHECK_CUDA_ERROR(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, const char* func, const char* file, int line) {
    if (result) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result)
                  << " \"" << func << "\" " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Function to initialize matrices with consistent values across data types
template <typename T>
void initialize_matrix(T* matrix, const int* values, unsigned int n) {
    for (unsigned int i = 0; i < n * n; ++i) {
        matrix[i] = static_cast<T>(values[i]);
    }
}

int main(int argc, char **argv) {
    // Check command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n block_dim" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    unsigned int n = std::atoi(argv[1]);
    unsigned int block_dim = std::atoi(argv[2]);
    
    // Allocate host memory for source values (integers)
    int* values = new int[n * n];
    
    // Generate random values with fixed seed for reproducibility
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(-10, 10);
    for (unsigned int i = 0; i < n * n; ++i) {
        values[i] = dist(gen);
    }
    
    // ===== Test for int matrices =====
    {
        // Allocate host memory
        int *h_A = new int[n * n];
        int *h_B = new int[n * n];
        int *h_C = new int[n * n];
        
        // Initialize matrices
        initialize_matrix(h_A, values, n);
        initialize_matrix(h_B, values, n);
        
        // Allocate device memory
        int *d_A, *d_B, *d_C;
        CHECK_CUDA_ERROR(cudaMalloc(&d_A, n * n * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_B, n * n * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_C, n * n * sizeof(int)));
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, n * n * sizeof(int), cudaMemcpyHostToDevice));
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        // Start timer
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        
        // Call the matrix multiplication function for int
        matmul_1(d_A, d_B, d_C, n, block_dim);
        
        // Stop timer
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        // Calculate elapsed time
        float milliseconds = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Print the first and last elements of the result
        std::cout << h_C[0] << std::endl;
        std::cout << h_C[n * n - 1] << std::endl;
        std::cout << milliseconds << std::endl;
        
        // Free device memory
        CHECK_CUDA_ERROR(cudaFree(d_A));
        CHECK_CUDA_ERROR(cudaFree(d_B));
        CHECK_CUDA_ERROR(cudaFree(d_C));
        
        // Destroy CUDA events
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
        
        // Free host memory
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }
    
    // ===== Test for float matrices =====
    {
        // Allocate host memory
        float *h_A = new float[n * n];
        float *h_B = new float[n * n];
        float *h_C = new float[n * n];
        
        // Initialize matrices
        initialize_matrix(h_A, values, n);
        initialize_matrix(h_B, values, n);
        
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        CHECK_CUDA_ERROR(cudaMalloc(&d_A, n * n * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_B, n * n * sizeof(float)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_C, n * n * sizeof(float)));
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice));
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        // Start timer
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        
        // Call the matrix multiplication function for float
        matmul_2(d_A, d_B, d_C, n, block_dim);
        
        // Stop timer
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        // Calculate elapsed time
        float milliseconds = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Print the first and last elements of the result
        std::cout << h_C[0] << std::endl;
        std::cout << h_C[n * n - 1] << std::endl;
        std::cout << milliseconds << std::endl;
        
        // Free device memory
        CHECK_CUDA_ERROR(cudaFree(d_A));
        CHECK_CUDA_ERROR(cudaFree(d_B));
        CHECK_CUDA_ERROR(cudaFree(d_C));
        
        // Destroy CUDA events
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
        
        // Free host memory
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }
    
    // ===== Test for double matrices =====
    {
        // Allocate host memory
        double *h_A = new double[n * n];
        double *h_B = new double[n * n];
        double *h_C = new double[n * n];
        
        // Initialize matrices
        initialize_matrix(h_A, values, n);
        initialize_matrix(h_B, values, n);
        
        // Allocate device memory
        double *d_A, *d_B, *d_C;
        CHECK_CUDA_ERROR(cudaMalloc(&d_A, n * n * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_B, n * n * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_C, n * n * sizeof(double)));
        
        // Copy data to device
        CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, n * n * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, n * n * sizeof(double), cudaMemcpyHostToDevice));
        
        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        // Start timer
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        
        // Call the matrix multiplication function for double
        matmul_3(d_A, d_B, d_C, n, block_dim);
        
        // Stop timer
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        // Calculate elapsed time
        float milliseconds = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // Copy result back to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, n * n * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Print the first and last elements of the result
        std::cout << h_C[0] << std::endl;
        std::cout << h_C[n * n - 1] << std::endl;
        std::cout << milliseconds << std::endl;
        
        // Free device memory
        CHECK_CUDA_ERROR(cudaFree(d_A));
        CHECK_CUDA_ERROR(cudaFree(d_B));
        CHECK_CUDA_ERROR(cudaFree(d_C));
        
        // Destroy CUDA events
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
        
        // Free host memory
        delete[] h_A;
        delete[] h_B;
        delete[] h_C;
    }
    
    // Free source values
    delete[] values;
    
    return 0;
}