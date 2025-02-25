#include "scan.h"

// Implement an inclusive scan function.
// out[0] = in[0]
// out[i] = out[i-1] + in[i]  for i = 1..(n-1)
void scan(const float *in, float *out, std::size_t n)
{
    if (n <= 0)
        return;

    // First element is just copied
    out[0] = in[0];

    // Compute the inclusive prefix sums
    for (std::size_t i = 1; i < n; i++)
    {
        out[i] = out[i - 1] + in[i];
    }
}
