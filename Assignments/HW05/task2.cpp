#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

__global__ void computeKernel(int *dA, int a) {
    // Compute the global thread index: 0 <= idx < 16
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = threadIdx.x;
    int y = blockIdx.x;
    dA[idx] = a * x + y;
}

int main() {
    // Seed the random number generator and generate a random integer 'a'
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    int a = std::rand() % 10; // Random integer in the range [0, 9]
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
