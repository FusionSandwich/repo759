// File: task3.cpp
#include "matmul.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

int main()
{
    // For the assignment, n should be >= 1000. You can change this if desired.
    const unsigned int n = 1000;

    // Print the number of rows first (per instructions)
    std::cout << n << std::endl;

    // Create random engine
    std::mt19937_64 rng(12345); // fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Allocate and fill A and B (row-major)
    std::vector<double> A(n * n), B(n * n);
    for (unsigned int i = 0; i < n * n; ++i)
    {
        A[i] = dist(rng);
        B[i] = dist(rng);
    }

    // Prepare arrays C1, C2, C3, C4
    std::vector<double> C1(n * n), C2(n * n), C3(n * n), C4(n * n);

    // mmul1
    {
        auto start = std::chrono::high_resolution_clock::now();
        mmul1(A.data(), B.data(), C1.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << time_ms << std::endl;       // time
        std::cout << C1[n * n - 1] << std::endl; // last element
    }

    // mmul2
    {
        auto start = std::chrono::high_resolution_clock::now();
        mmul2(A.data(), B.data(), C2.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << time_ms << std::endl;       // time
        std::cout << C2[n * n - 1] << std::endl; // last element
    }

    // mmul3
    {
        auto start = std::chrono::high_resolution_clock::now();
        mmul3(A.data(), B.data(), C3.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << time_ms << std::endl;       // time
        std::cout << C3[n * n - 1] << std::endl; // last element
    }

    // mmul4
    {
        auto start = std::chrono::high_resolution_clock::now();
        mmul4(A, B, C4.data(), n);
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << time_ms << std::endl;       // time
        std::cout << C4[n * n - 1] << std::endl; // last element
    }

    return 0;
}
