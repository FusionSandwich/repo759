// convolution.cpp
#include "convolution.h"
#include <omp.h>

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m) {
    std::size_t radius = m / 2;
    // Parallelize the outer two loops; collapse both for better load balance.
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < n; i++) {
        for (std::size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            // Convolve the mask over the image at position (i, j)
            for (std::size_t mi = 0; mi < m; mi++) {
                for (std::size_t mj = 0; mj < m; mj++) {
                    long ii = static_cast<long>(i) + static_cast<long>(mi) - static_cast<long>(radius);
                    long jj = static_cast<long>(j) + static_cast<long>(mj) - static_cast<long>(radius);
                    // Use zero-padding for out-of-bound indices.
                    if (ii >= 0 && ii < static_cast<long>(n) && jj >= 0 && jj < static_cast<long>(n)) {
                        sum += image[ii * n + jj] * mask[mi * m + mj];
                    }
                }
            }
            output[i * n + j] = sum;
        }
    }
}
