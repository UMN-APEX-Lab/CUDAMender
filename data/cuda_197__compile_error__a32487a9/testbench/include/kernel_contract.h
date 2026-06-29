#ifndef KERNEL_CONTRACT_H
#define KERNEL_CONTRACT_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) {                                     \
    cudaError_t error = call;                                  \
    if (error != cudaSuccess) {                                \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

// This must stay as 32.
constexpr int SEGMENT_SIZE = 32;

__global__ void k_compactElementsOfSegmentsWithThreshold(
    int numSegments, 
    float* array_d, 
    float threshold, 
    float defaultValue
);

#endif // KERNEL_CONTRACT_H