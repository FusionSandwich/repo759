#include "msort.h"
#include <algorithm>  // for std::sort
#include <omp.h>
#include <vector>

// Merge two sorted subarrays [left, mid) and [mid, right) into arr
static void merge(int* arr, std::size_t left, std::size_t mid, std::size_t right)
{
    std::size_t n1 = mid - left;
    std::size_t n2 = right - mid;
    
    // Create temporary vectors to hold the two halves
    std::vector<int> L(n1);
    std::vector<int> R(n2);
    
    for (std::size_t i = 0; i < n1; i++) {
        L[i] = arr[left + i];
    }
    for (std::size_t j = 0; j < n2; j++) {
        R[j] = arr[mid + j];
    }
    
    std::size_t i = 0, j = 0, k = left;
    // Merge the two halves back into arr
    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }
    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
}

// Recursive helper function for parallel merge sort
static void parallel_mergesort(int* arr, std::size_t left, std::size_t right, const std::size_t threshold)
{
    if (right - left <= 1)
        return;  // Base case: array of size 0 or 1 is already sorted.
    
    std::size_t size = right - left;
    if (size <= threshold) {
        // Use serial sort when below threshold to avoid task overhead.
        std::sort(arr + left, arr + right);
        return;
    }
    
    std::size_t mid = left + size / 2;
    
    // Spawn two tasks to sort the halves in parallel.
    #pragma omp task shared(arr)
    {
        parallel_mergesort(arr, left, mid, threshold);
    }
    #pragma omp task shared(arr)
    {
        parallel_mergesort(arr, mid, right, threshold);
    }
    #pragma omp taskwait  // Wait for both tasks to finish.
    
    // Merge the two sorted halves.
    merge(arr, left, mid, right);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold)
{
    // Create a parallel region and a single task to kick-start the recursion.
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallel_mergesort(arr, 0, n, threshold);
        }
    }
}
