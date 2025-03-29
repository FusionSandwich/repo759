#include <cstdio>
#include <cuda_runtime.h>

__global__ void factorialKernel() {
    // Each thread computes the factorial of (threadIdx.x + 1)
    int n = threadIdx.x + 1;  // Thread 0 computes 1!, thread 1 computes 2!, etc.
    int factorial = 1;
    for (int i = 1; i <= n; ++i) {
        factorial *= i;
    }
    // Print in the format "n!=factorial"
    printf("%d!=%d\n", n, factorial);
}

int main() {
    // Launch a kernel with 1 block and 8 threads
    factorialKernel<<<1, 8>>>();
    // Wait for the kernel to finish before exiting main
    cudaDeviceSynchronize();
    return 0;
}
