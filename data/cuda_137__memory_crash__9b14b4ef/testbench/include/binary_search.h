#ifndef BINARY_SEARCH_H
#define BINARY_SEARCH_H

#include <cuda_runtime.h>

// Constants
#define THREADS_PER_BLOCK 256
#define SHARED_MEM_SIZE 256

__global__ void k_binarySearch(int *arr_d, int *queries_d, int *results_d, int arrSize, int querySize);

#endif // BINARY_SEARCH_H