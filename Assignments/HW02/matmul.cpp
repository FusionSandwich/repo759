// File: matmul.cpp
#include "matmul.h"

void mmul1(const double *A, const double *B, double *C, const unsigned int n)
{
    // Zero-initialize C to ensure no leftover data
    for (unsigned int idx = 0; idx < n * n; ++idx)
    {
        C[idx] = 0.0;
    }

    // Triple nested loop in (i, j, k) order
    for (unsigned int i = 0; i < n; ++i)
    {
        for (unsigned int j = 0; j < n; ++j)
        {
            for (unsigned int k = 0; k < n; ++k)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul2(const double *A, const double *B, double *C, const unsigned int n)
{
    // Zero-initialize C
    for (unsigned int idx = 0; idx < n * n; ++idx)
    {
        C[idx] = 0.0;
    }

    // Triple nested loop in (i, k, j) order
    for (unsigned int i = 0; i < n; ++i)
    {
        for (unsigned int k = 0; k < n; ++k)
        {
            for (unsigned int j = 0; j < n; ++j)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul3(const double *A, const double *B, double *C, const unsigned int n)
{
    // Zero-initialize C
    for (unsigned int idx = 0; idx < n * n; ++idx)
    {
        C[idx] = 0.0;
    }

    // Triple nested loop in (j, k, i) order
    // (outermost is j, then k, then i).
    for (unsigned int j = 0; j < n; ++j)
    {
        for (unsigned int k = 0; k < n; ++k)
        {
            for (unsigned int i = 0; i < n; ++i)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void mmul4(const std::vector<double> &A, const std::vector<double> &B,
           double *C, const unsigned int n)
{
    // Zero-initialize C
    for (unsigned int idx = 0; idx < n * n; ++idx)
    {
        C[idx] = 0.0;
    }

    // Same loop ordering as mmul1, but A & B are passed in as vectors
    for (unsigned int i = 0; i < n; ++i)
    {
        for (unsigned int j = 0; j < n; ++j)
        {
            for (unsigned int k = 0; k < n; ++k)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}
