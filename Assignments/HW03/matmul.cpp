#include "matmul.h"

void mmul(const float* A, const float* B, float* C, const std::size_t n)
{
    // 1) Zero-initialize matrix C
    #pragma omp parallel for
    for (std::size_t idx = 0; idx < n*n; ++idx) {
        C[idx] = 0.0f;
    }

    // 2) Perform parallel matrix multiply in (i, k, j) order
    //    This corresponds to the mmul2 pattern from HW02 task3.
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t k = 0; k < n; ++k)
        {
            float temp = A[i*n + k];
            for (std::size_t j = 0; j < n; ++j)
            {
                C[i*n + j] += temp * B[k*n + j];
            }
        }
    }
}
