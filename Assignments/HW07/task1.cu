// File: task1.cu
// Test program for tiled matrix multiplication functions.

#include <iostream>
#include <vector>
#include <cstdlib> // For atoi, EXIT_FAILURE, EXIT_SUCCESS
#include <cuda_runtime.h>
#include "matmul.cuh" // Includes the declarations for matmul_1, matmul_2, matmul_3

// Helper function to check CUDA errors
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

// Template function to initialize matrices
template <typename T>
void initializeMatrices(T* A, T* B, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // Simple initialization for demonstration
            // In a real test, you might use random numbers or other patterns
            if constexpr (std::is_integral_v<T>) {
                A[i * n + j] = static_cast<T>(i + j + 1); // Example integer init
                B[i * n + j] = static_cast<T>(i - j + 1); // Example integer init
            } else {
                A[i * n + j] = static_cast<T>(i + j + 1.1); // Example floating point init
                B[i * n + j] = static_cast<T>(i - j + 1.1); // Example floating point init
            }
        }
    }
}


// Template function to run matrix multiplication test
template <typename T>
void runMatmulTest(const char* typeName, void (*matmul_func)(const T*, const T*, T*, unsigned int, unsigned int),
                   unsigned int n, unsigned int block_dim) {
    size_t matrix_size = (size_t)n * n; // Use size_t for potentially large sizes
    size_t bytes = matrix_size * sizeof(T);

    // Allocate managed memory for A, B, C
    // Managed memory is accessible from both host and device
    T *A_managed, *B_managed, *C_managed;
    checkCuda(cudaMallocManaged(&A_managed, bytes));
    checkCuda(cudaMallocManaged(&B_managed, bytes));
    checkCuda(cudaMallocManaged(&C_managed, bytes)); // C will store the result

    // Initialize matrices A and B on the host using the managed pointers
    initializeMatrices(A_managed, B_managed, n);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start));
    checkCuda(cudaEventCreate(&stop));

    // Record start event
    checkCuda(cudaEventRecord(start));

    // Call the specific matrix multiplication function
    matmul_func(A_managed, B_managed, C_managed, n, block_dim);

    // Record stop event
    checkCuda(cudaEventRecord(stop));

    // Wait for the stop event to complete
    checkCuda(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop));

    // Print results: first element, last element, time
    // Accessing managed memory directly from host after device sync
    if (matrix_size > 0) {
         std::cout << C_managed[0] << std::endl;
         std::cout << C_managed[matrix_size - 1] << std::endl;
    } else {
         std::cout << "Matrix size is 0" << std::endl;
         std::cout << "Matrix size is 0" << std::endl;
    }
    std::cout << milliseconds << std::endl;


    // Cleanup
    checkCuda(cudaEventDestroy(start));
    checkCuda(cudaEventDestroy(stop));
    checkCuda(cudaFree(A_managed));
    checkCuda(cudaFree(B_managed));
    checkCuda(cudaFree(C_managed));
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " n block_dim" << std::endl;
        return EXIT_FAILURE;
    }

    // Parse command line arguments
    unsigned int n = atoi(argv[1]);
    unsigned int block_dim = atoi(argv[2]);

    if (n <= 0 || block_dim <= 0) {
         std::cerr << "Error: n and block_dim must be positive integers." << std::endl;
         return EXIT_FAILURE;
    }
     if (block_dim > 32) {
         // Based on the static shared memory allocation in matmul.cu
         // Adjust if matmul.cu changes (e.g., uses dynamic shared memory)
         std::cerr << "Warning: block_dim > 32 may exceed static shared memory allocation in matmul_kernel." << std::endl;
         // Consider adding a hard exit if required by assignment constraints or kernel implementation.
     }


    // Run tests for int, float, and double [cite: 14]
    runMatmulTest<int>("int", matmul_1, n, block_dim);
    runMatmulTest<float>("float", matmul_2, n, block_dim);
    runMatmulTest<double>("double", matmul_3, n, block_dim);

    return EXIT_SUCCESS;
}