#ifndef KERNEL_CONTRACT_H
#define KERNEL_CONTRACT_H

#include <cuda_runtime.h>
#include <cstdint>

// Type of data to use for coalesced load.
using INPUT_CHUNK_TYPE = int4;
// Number of elements to be used in same calculation.
constexpr int CHUNK_ELEMENTS = sizeof(INPUT_CHUNK_TYPE) / sizeof(int32_t);

__global__ void k_calculateDifferencesFromPowerOfAveragePerChunk(
    int32_t * input_d, 
    float * output_d, 
    int numChunks);

#endif // KERNEL_CONTRACT_H