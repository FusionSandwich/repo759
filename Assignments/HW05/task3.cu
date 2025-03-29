#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "vscale.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <n>\n";
        return 1;
    }

    // 1. Parse n from the command line
    unsigned int n = static_cast<unsigned int>(std::stoi(argv[1]));
    std::cout << "n = " << n << std::endl;

    // 2. Create and fill two arrays of length n
    //    hA in [-10.0, 10.0], hB in [0.0, 1.0]
    float* hA = new float[n];
    float* hB = new float[n];

    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (unsigned int i = 0; i < n; ++i) {
        // random float in [-10, 10]
        float randA = (static_cast<float>(std::rand()) / RAND_MAX) * 20.0f - 10.0f;
        // random float in [0, 1]
        float randB = (static_cast<float>(std::rand()) / RAND_MAX);

        hA[i] = randA;
        hB[i] = randB;
    }

    // 3. Allocate device memory and copy hA, hB to device
    float *dA, *dB;
    cudaMalloc(&dA, n * sizeof(float));
    cudaMalloc(&dB, n * sizeof(float));
    cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

    // 4. Prepare CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 5. Invoke the vscale kernel
    //    We'll use 512 threads per block by default
    unsigned int threadsPerBlock = 512;
    unsigned int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);
    vscale<<<numBlocks, threadsPerBlock>>>(dA, dB, n);
    cudaEventRecord(stop);

    // 6. Wait for the kernel to finish and compute elapsed time
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 7. Copy results back to host
    cudaMemcpy(hB, dB, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 8. Print the time, plus the first and last elements
    std::cout << milliseconds << " ms\n";
    std::cout << hB[0] << "\n";
    std::cout << hB[n - 1] << "\n";

    // 9. Free resources
    cudaFree(dA);
    cudaFree(dB);
    delete[] hA;
    delete[] hB;

    return 0;
}
