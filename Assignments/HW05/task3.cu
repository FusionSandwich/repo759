#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "vscale.cuh"

int main(int argc, char* argv[]) {
    // Open the output file in truncation mode to ensure a clean start.
    std::ofstream outFile("results.txt", std::ofstream::out | std::ofstream::trunc);
    if (!outFile) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }
    // Write header line.
    outFile << "n time_512 time_16 first_512 last_512 first_16 last_16\n";

    // Seed the random number generator.
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Loop over exponents 10 to 29 (n = 2^exp).
    for (int exp = 10; exp <= 29; exp++) {
        unsigned int n = 1u << exp;  // n = 2^exp

        // Allocate and initialize host arrays.
        float* hA = new float[n];
        float* hB = new float[n];
        for (unsigned int i = 0; i < n; ++i) {
            // hA: random float in [-10, 10]
            hA[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 20.0f - 10.0f;
            // hB: random float in [0, 1]
            hB[i] = static_cast<float>(std::rand()) / RAND_MAX;
        }

        // Allocate device memory for a and b.
        float *dA, *dB;
        cudaMalloc(&dA, n * sizeof(float));
        cudaMalloc(&dB, n * sizeof(float));

        // Copy hA and hB to device.
        cudaMemcpy(dA, hA, n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

        // ---- Experiment 1: Using 512 threads per block ----
        unsigned int threads512 = 512;
        unsigned int numBlocks512 = (n + threads512 - 1) / threads512;

        cudaEvent_t start512, stop512;
        cudaEventCreate(&start512);
        cudaEventCreate(&stop512);

        cudaEventRecord(start512);
        vscale<<<numBlocks512, threads512>>>(dA, dB, n);
        cudaEventRecord(stop512);
        cudaEventSynchronize(stop512);
        float time512 = 0.0f;
        cudaEventElapsedTime(&time512, start512, stop512);

        // Copy results from device.
        float *result512 = new float[n];
        cudaMemcpy(result512, dB, n * sizeof(float), cudaMemcpyDeviceToHost);
        float first512 = result512[0];
        float last512 = result512[n - 1];

        cudaEventDestroy(start512);
        cudaEventDestroy(stop512);

        // ---- Experiment 2: Using 16 threads per block ----
        // Restore original hB values into device array dB.
        cudaMemcpy(dB, hB, n * sizeof(float), cudaMemcpyHostToDevice);

        unsigned int threads16 = 16;
        unsigned int numBlocks16 = (n + threads16 - 1) / threads16;

        cudaEvent_t start16, stop16;
        cudaEventCreate(&start16);
        cudaEventCreate(&stop16);

        cudaEventRecord(start16);
        vscale<<<numBlocks16, threads16>>>(dA, dB, n);
        cudaEventRecord(stop16);
        cudaEventSynchronize(stop16);
        float time16 = 0.0f;
        cudaEventElapsedTime(&time16, start16, stop16);

        float *result16 = new float[n];
        cudaMemcpy(result16, dB, n * sizeof(float), cudaMemcpyDeviceToHost);
        float first16 = result16[0];
        float last16 = result16[n - 1];

        cudaEventDestroy(start16);
        cudaEventDestroy(stop16);

        // Write one clean line of results for this n.
        outFile << n << " " << time512 << " " << time16 << " " 
                << first512 << " " << last512 << " " 
                << first16 << " " << last16 << "\n";

        // Cleanup for this iteration.
        cudaFree(dA);
        cudaFree(dB);
        delete[] hA;
        delete[] hB;
        delete[] result512;
        delete[] result16;
    }

    outFile.close();
    return 0;
}
