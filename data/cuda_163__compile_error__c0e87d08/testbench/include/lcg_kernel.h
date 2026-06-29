#ifndef LCG_KERNEL_H
#define LCG_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <device_launch_parameters.h>

// Algorithm-related constants.
#define MAXIMUM_ARRAY_LENGTH 100000
// Number of 32bit elements per access per thread. The data array will be zero-padded accordingly with this number.
#define MEMORY_ACCESS_WIDTH (sizeof(uint4) / sizeof(uint32_t))
#define MAXIMUM_ARRAY_LENGTH_WITH_PADDING_FOR_WIDE_ACCESS ((1 + (MAXIMUM_ARRAY_LENGTH - 1) / MEMORY_ACCESS_WIDTH) * MEMORY_ACCESS_WIDTH)
#define MAXIMUM_BUFFER_BYTES (sizeof(uint32_t) * MAXIMUM_ARRAY_LENGTH_WITH_PADDING_FOR_WIDE_ACCESS)
// Constants for the linear congruential generator algorithm.
#define LCG_MULTIPLIER (134775813ull)
#define LCG_OFFSET (1ull)
#define LCG_MODULUS (4294967291ull)
#define LCG_STEPS 100

// CUDA-related constants.
#define NUMBER_OF_SHARED_MEMORY_ARRAYS_PER_BLOCK 2
#define MAXIMUM_NUMBER_OF_THREADS_PER_BLOCK_ALLOWED 1024

__global__ void k_calculateLCG(uint32_t * __restrict__ data_d, int numElements);

#endif // LCG_KERNEL_H