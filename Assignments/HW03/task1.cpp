#include <iostream>
#include <cstdlib>
#include <chrono>
#include <omp.h>     // for omp_set_num_threads
#include "matmul.h"  // your parallel mmul signature

int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <matrix-size> <num-threads>\n";
        return 1;
    }

    std::size_t n = static_cast<std::size_t>(std::atoi(argv[1]));
    int t = std::atoi(argv[2]);

    // Set the desired number of threads for OpenMP
    omp_set_num_threads(t);

    // Allocate memory for A, B, and C
    float* A = new float[n*n];
    float* B = new float[n*n];
    float* C = new float[n*n];

    // Initialize A and B
    // (Example: fill them with some values, e.g. i+1, random, etc.)
    for (std::size_t i = 0; i < n*n; ++i) {
        A[i] = static_cast<float>(i + 1);
        B[i] = static_cast<float>(i + 2);
    }

    // Measure time
    auto start = std::chrono::steady_clock::now();

    // Call parallel mmul
    mmul(A, B, C, n);

    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = 
        std::chrono::duration<double, std::milli>(end - start).count();

    // Print the first and last elements of C
    std::cout << C[0] << std::endl;
    std::cout << C[n*n - 1] << std::endl;
    // Print elapsed time in milliseconds
    std::cout << elapsed_ms << std::endl;

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
