// File: task2.cu
#include "reduce.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr,
            "Usage: %s <N (<=2^30)> <threads_per_block (power of 2)>\n",
            argv[0]);
        return EXIT_FAILURE;
    }

    unsigned int N = static_cast<unsigned int>(atoll(argv[1]));
    unsigned int threads_per_block = static_cast<unsigned int>(atoi(argv[2]));

    // 1) Host allocation & init
    float *h_data = (float*)malloc(N * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "Host malloc failed\n");
        return EXIT_FAILURE;
    }
    srand((unsigned)time(nullptr));
    for (unsigned int i = 0; i < N; i++) {
        h_data[i] = 2.0f * rand() / RAND_MAX - 1.0f;  // in [-1,1]
    }

    // 2) Device allocations
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int blocks = (N + threads_per_block * 2 - 1)
                              / (threads_per_block * 2);
    cudaMalloc(&d_out, blocks * sizeof(float));

    // 3) Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 4) Call reduction
    cudaEventRecord(start);
    reduce(&d_in, &d_out, N, threads_per_block);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // 5) Copy back and print
    float result = 0.0f;
    cudaMemcpy(&result, d_in, sizeof(float), cudaMemcpyDeviceToHost);
    printf("%f\n", result);
    printf("%f\n", ms);

    // 6) Cleanup
    free(h_data);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
