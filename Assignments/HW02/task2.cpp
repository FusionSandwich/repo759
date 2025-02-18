#include <iostream>
#include <cstdlib> // for std::rand, std::srand, std::time
#include <ctime>   // for std::time
#include <chrono>  // for high_resolution_clock
#include "convolution.h"

int main(int argc, char *argv[])
{
    // We expect exactly 2 command-line arguments: n (image size), m (mask size).
    // If they're missing, print usage info and exit.
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " n m\n";
        return 1;
    }

    // Convert command-line arguments to size_t.
    // n is the dimension of the image (n x n),
    // m is the dimension of the mask (m x m).
    std::size_t n = std::stoul(argv[1]);
    std::size_t m = std::stoul(argv[2]);

    // According to the assignment, m must be an odd number.
    // If it's even, report an error and exit.
    if (m % 2 == 0)
    {
        std::cerr << "Error: m must be an odd number.\n";
        return 1;
    }

    // Dynamically allocate memory for:
    //  1) the image (n*n floats)
    //  2) the mask (m*m floats)
    //  3) the output (n*n floats to store the convolved result)
    float *image = new float[n * n];
    float *mask = new float[m * m];
    float *output = new float[n * n];

    // Seed the random number generator with the current time.
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Fill 'image' with random float numbers in the range [-10, 10].
    // That is: rand()/RAND_MAX generates [0,1], which we scale and shift.
    for (std::size_t i = 0; i < n * n; ++i)
    {
        image[i] = static_cast<float>(std::rand()) / RAND_MAX * 20.0f - 10.0f;
    }

    // Fill 'mask' with random float numbers in the range [-1, 1].
    for (std::size_t i = 0; i < m * m; ++i)
    {
        mask[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    // Record the time before calling our convolution function.
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the convolution of the image with the mask and store in 'output'.
    convolve(image, output, n, mask, m);

    // Record the time again after convolution completes.
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time in milliseconds for the convolution call.
    std::chrono::duration<double, std::milli> duration = end - start;

    // Print the time taken to perform the convolution (in ms).
    std::cout << duration.count() << "\n";

    // Print the first element of the convolved output.
    // This gives a quick sanity check of the results.
    std::cout << output[0] << "\n";

    // Print the last element of the convolved output.
    // Another quick check to ensure the data looks as expected.
    std::cout << output[n * n - 1] << "\n";

    // Clean up the dynamically allocated arrays to avoid memory leaks.
    delete[] image;
    delete[] mask;
    delete[] output;

    return 0;
}