#include "bitonic_sort.h"
#include "common.h"
#include <cfloat>

__global__ void k_bitonicSort(float* inputData, int size) {
    int threadId = threadIdx.x + blockIdx.x * blockDim.x;
    int localIdx = threadIdx.x;

    // Shared memory array to store part of the data for sorting
    extern __shared__ float sharedData[];

    // Each thread loads data into shared memory
    sharedData[localIdx] = (threadId < size) ? inputData[threadId] : FLT_MAX;
    __syncthreads();

    // Perform Bitonic Sort using shared memory
    for (int stage = 2; stage <= BLOCK_SIZE; stage *= 2) {
        for (int stride = stage / 2; stride > 0; stride /= 2) {
            // Compute comparison indices
            int compareIdx = localIdx ^ stride;

            // Ensure threads stay within bounds
            if (compareIdx > localIdx) {
                // Sorting direction depends on stage
                bool ascending = ((localIdx & stage) == 0);

                // Compare and swap
                if (ascending == (sharedData[localIdx] > sharedData[compareIdx])) {
                    float temp = sharedData[localIdx];
                    sharedData[localIdx] = sharedData[compareIdx];
                    sharedData[compareIdx] = temp;
                }
            }
            __syncthreads(); // Synchronize after each stride
        }
    }

    // Write the sorted data back to global memory
    if (threadId < size) {
        inputData[threadId] = sharedData[localIdx];
    }
}