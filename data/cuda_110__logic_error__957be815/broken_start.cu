#include "include/bitonic_sort.h"
#include "include/common.h"

__global__ void k_bitonicSort(float* inputData, int size) {
    // Bitonic sort typically requires size to be a power of 2.
    // For a general implementation within a single kernel as requested,
    // we use the standard bitonic sorting network logic.
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    
    // Load data into shared memory
    if (tid < size) {
        sdata[tid] = inputData[tid];
    }
    __syncthreads();

    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid && tid < size && ixj < size) {
                bool ascending = ((tid & k) == 0);
                float a = sdata[tid];
                float b = sdata[ixj];

                if ((a > b) == ascending) {
                    sdata[tid] = b;
                    sdata[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    if (tid < size) {
        inputData[tid] = sdata[tid];
    }
}