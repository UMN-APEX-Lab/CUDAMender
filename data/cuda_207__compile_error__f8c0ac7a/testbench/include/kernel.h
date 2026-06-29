#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>
#include <cstdint>

// Each thread loads this number of integers from global memory simultaneously. Since each integer is 32 bits, each thread processes 128 bits at a time before loading new data.
constexpr uint32_t FOUR_INTEGERS_PER_THREAD = 4;

// Computes a sliding window of 3-input XOR (where the window consists of the XOR inputs) on all bits in the integersIn_d array and writes the output to the integersOut_d array.
__global__ void k_calculateElement(uint32_t booleanElementsPerInteger, uint32_t numIntegers, uint32_t* integersIn_d, uint32_t* integersOut_d);

#endif // KERNEL_H