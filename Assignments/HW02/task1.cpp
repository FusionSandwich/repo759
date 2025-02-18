#include <iostream>
#include <chrono>  // for high_resolution_clock
#include <random>  // for mt19937, uniform_real_distribution
#include <cstdlib> // for std::atoi
#include "scan.h"

// Program to benchmark inclusive scan performance
int main(int argc, char *argv[])
{
    // Check for command line arguments
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " n\n";
        return 1;
    }

    // Read n from command line
    int n = std::atoi(argv[1]);
    if (n <= 0)
    {
        std::cerr << "Error: n must be a positive integer.\n";
        return 1;
    }

    // Allocate arrays
    float *input = new float[n];
    float *output = new float[n];

    // Initialize random number generation in range [-1.0, 1.0]
    // Using this method 1.0 is not included, but -1.0 is included
    // Told this is okay in the assignment
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Fill input array with random numbers
    for (int i = 0; i < n; i++)
    {
        input[i] = dist(rng);
    }

    // Start timing scan
    auto start = std::chrono::high_resolution_clock::now();

    // Perform inclusive scan
    // (Implementation details are in scan.h / scan.cpp).
    scan(input, output, n);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double ms = elapsed.count();

    // Print timing
    std::cout << ms << "\n"; // Print time in milliseconds

    // Print the first and last elements of the scanned array
    std::cout << output[0] << "\n";     // Print first element
    std::cout << output[n - 1] << "\n"; // Print last element

    // Deallocate memory
    delete[] input;
    delete[] output;

    return 0;
}
