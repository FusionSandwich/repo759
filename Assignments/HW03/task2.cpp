// task2.cpp
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "convolution.h"
#include <omp.h>

int main(int argc, char* argv[]) {
    // Two usage cases:
    // 1) ./task2 <n> <t>        -> uses a default 3×3 mask.
    // 2) ./task2 <n> <m> <t>    -> uses an m×m mask.
    if (argc < 3) {
        std::cerr << "Usage:\n";
        std::cerr << "  " << argv[0] << " <n> <t>        # for a 3×3 mask\n";
        std::cerr << "  " << argv[0] << " <n> <m> <t>    # for an m×m mask\n";
        return 1;
    }

    std::size_t n = std::stoul(argv[1]);
    std::size_t maskSize = 3; // default mask size
    int threadCount;

    if (argc == 3) {
        threadCount = std::atoi(argv[2]);
    } else {
        maskSize = std::stoul(argv[2]);
        threadCount = std::atoi(argv[3]);
    }

    // Set the number of OpenMP threads
    omp_set_num_threads(threadCount);

    // Create and fill an n×n image with random floats in [-10.0, 10.0]
    float* image = new float[n * n];
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (std::size_t i = 0; i < n * n; ++i) {
        image[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 20.0f - 10.0f;
    }

    // Create and fill the mask with random floats in [-1.0, 1.0]
    float* mask = new float[maskSize * maskSize];
    for (std::size_t i = 0; i < maskSize * maskSize; ++i) {
        mask[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f - 1.0f;
    }

    // Allocate output array (n×n)
    float* output = new float[n * n];

    // Time the convolution operation in milliseconds.
    auto start = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, maskSize);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Print the first element, the last element, and the elapsed time.
    std::cout << output[0] << std::endl;
    std::cout << output[n * n - 1] << std::endl;
    std::cout << elapsed.count() << std::endl;

    // Free allocated memory.
    delete[] image;
    delete[] mask;
    delete[] output;
    return 0;
}
