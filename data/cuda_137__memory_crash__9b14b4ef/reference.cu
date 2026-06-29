// file: solution.cu
#include "binary_search.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void k_binarySearch(int *arr_d, int *queries_d, int *results_d, int arrSize, int querySize) {
    cg::grid_group grid = cg::this_grid(); 

    int globalThreadId = threadIdx.x + blockIdx.x * blockDim.x;
    int gridStride = blockDim.x * gridDim.x;
    int threadIndex = threadIdx.x;
    int threadsPerBlock = blockDim.x;

    __shared__ int sharedArr[SHARED_MEM_SIZE];

    // Cooperatively load first portion of array into shared memory
    for (int i = threadIndex; i < arrSize && i < SHARED_MEM_SIZE; i += threadsPerBlock) {
        sharedArr[i] = arr_d[i];
    }

    __syncthreads();

    // Process multiple queries using grid-stride loop
    for (int i = globalThreadId; i < querySize; i += gridStride) {
        int target = queries_d[i];
        int left = 0;
        int right = arrSize - 1;
        int result = -1;

        // Binary search
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int midValue;
            
            // Use shared memory when possible
            if(mid < SHARED_MEM_SIZE) {
                midValue = sharedArr[mid];
            } else {
                midValue = arr_d[mid];
            }

            if (midValue == target) {
                result = mid;
                break;
            }
            
            if (midValue < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        results_d[i] = result;
    }
}
