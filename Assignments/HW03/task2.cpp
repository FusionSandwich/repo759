#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <omp.h>
#include "convolution.h"

int main(int argc, char* argv[])
{
    // Check command-line arguments
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " n t" << std::endl;
        return 1;
    }

    // Parse arguments
    std::size_t n = std::stoul(argv[1]); // Image size (n x n)
    int t = std::stoi(argv[2]);          // Number of threads
    if (t < 1 || t > 20)
    {
        std::cerr << "Thread count must be between 1 and 20" << std::endl;
        return 1;
    }
    omp_set_num_threads(t);

    // Seed random number generator with current time for unique runs
    std::default_random_engine generator(static_cast<unsigned>(std::time(nullptr)));

    // 1. Create and fill the image matrix
    std::uniform_real_distribution<float> img_dist(-10.0f, 10.0f);
    std::vector<float> image(n * n);
    for (auto& val : image)
    {
        val = img_dist(generator); // Random floats between -10.0 and 10.0
    }

    // 2. Create and fill the 3x3 mask matrix
    std::size_t m = 3; // Fixed size for mask
    std::uniform_real_distribution<float> mask_dist(-1.0f, 1.0f);
    std::vector<float> mask(m * m);
    for (auto& val : mask)
    {
        val = mask_dist(generator); // Random floats between -1.0 and 1.0
    }

    // 3. Allocate output array for convolution result
    std::vector<float> output(n * n);

    // Measure execution time of convolution
    double start = omp_get_wtime();
    convolve(image.data(), output.data(), n, mask.data(), m);
    double end = omp_get_wtime();
    double time_ms = (end - start) * 1000.0;

    // Output results
    std::cout << output[0] << std::endl;         // First element of output
    std::cout << output[n * n - 1] << std::endl; // Last element of output
    std::cout << time_ms << std::endl;           // Time in milliseconds

    return 0;
}
