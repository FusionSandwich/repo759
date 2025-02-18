#include "convolution.h"
#include <cstdint>

// Applies convolution mask to image with zero-padded corners and one-padded edges
void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m)
{
    // Calculate center offset for odd-sized mask (m=3 → c=1)
    int c = (static_cast<int>(m) - 1) / 2;
    // Slide the mask over every pixel in the n x n image
    for (std::size_t x = 0; x < n; ++x) // Row index
    {
        for (std::size_t y = 0; y < n; ++y) // Column index
        {
            float sum = 0.0f; // Accumulator for convolution result
            for (std::size_t i = 0; i < m; ++i)
            {
                for (std::size_t j = 0; j < m; ++j)
                {
                    // Calculate image coordinates relative to mask center
                    int64_t di = static_cast<int64_t>(i) - c;
                    int64_t dj = static_cast<int64_t>(j) - c;
                    int64_t a = static_cast<int64_t>(x) + di;
                    int64_t b = static_cast<int64_t>(y) + dj;
                    // Check if image coordinates are within bounds
                    bool a_in = (a >= 0) && (a < static_cast<int64_t>(n));
                    bool b_in = (b >= 0) && (b < static_cast<int64_t>(n));
                    // Fetch image value or pad with 0 or 1
                    float val;
                    if (a_in && b_in)
                    {
                        val = image[static_cast<std::size_t>(a) * n + static_cast<std::size_t>(b)];
                    }
                    else
                    {
                        // Handle padding logic:
                        // 1 for edges (exactly 1 coordinate out-of-bounds), 0 for corners
                        int out_of_bounds = 0;
                        if (!a_in)
                            out_of_bounds++;
                        if (!b_in)
                            out_of_bounds++;
                        val = (out_of_bounds == 1) ? 1.0f : 0.0f;
                    }
                    sum += mask[i * m + j] * val;
                }
            }
            output[x * n + y] = sum;
        }
    }
}