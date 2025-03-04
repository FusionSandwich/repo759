#include <iostream>
#include <cstdlib>
#include <ctime>      // For time()
#include <chrono>
#include <omp.h>
#include "msort.h"

int main(int argc, char* argv[])
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " n t threshold\n";
        return 1;
    }
    
    // Parse command-line arguments.
    std::size_t n = std::stoul(argv[1]);        // Array size.
    int t         = std::stoi(argv[2]);           // Number of threads.
    std::size_t threshold = std::stoul(argv[3]);    // Threshold for parallel recursion.
    
    // Allocate and fill the array with random integers in the range [-1000, 1000].
    int* arr = new int[n];
    
    // Use the current time as the seed to produce a different sequence each run.
    std::srand(std::time(nullptr));
    for (std::size_t i = 0; i < n; i++) {
        arr[i] = (std::rand() % 2001) - 1000;  // Generates numbers between -1000 and 1000.
    }
    
    // Set the number of threads for OpenMP.
    omp_set_num_threads(t);
    
    // Time the merge sort execution.
    auto start = std::chrono::high_resolution_clock::now();
    msort(arr, n, threshold);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    
    // Print the first element, last element, and elapsed time.
    std::cout << arr[0] << "\n";
    std::cout << arr[n - 1] << "\n";
    std::cout << elapsed.count() << "\n";
    
    delete[] arr;
    return 0;
}
