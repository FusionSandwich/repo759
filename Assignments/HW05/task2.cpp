#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <random>

__global__ void computeKernel(int *dA, int a) {
    // Compute the global thread index: 0 <= idx < 16
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = threadIdx.x;
    int y = blockIdx.x;
    dA[idx] = a * x + y;
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 9);
    int a = dist(gen);
    std::cout << "Random a: " << a << "\n";
    const int numElements = 16;
    int *dA;
    // Allocate device memory for 16 integers
    cudaMalloc(&dA, numElements * sizeof(int));

    // Launch kernel with 2 blocks of 8 threads each (2 x 8 = 16 threads)
    computeKernel<<<2, 8>>>(dA, a);
    cudaDeviceSynchronize();

    // Copy the computed values back to the host
    int hA[numElements];
    cudaMemcpy(hA, dA, numElements * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the 16 values, separated by a single space
    for (int i = 0; i < numElements; ++i) {
        std::cout << hA[i] << " ";
    }
    std::cout << std::endl;

    // Clean up device memory
    cudaFree(dA);
    return 0;
}
