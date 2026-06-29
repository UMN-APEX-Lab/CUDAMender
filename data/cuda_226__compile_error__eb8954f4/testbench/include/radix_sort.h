#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cooperative_groups.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

// Core configuration constants (shared across host and device)
#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4
#define RADIX_BITS 4
#define RADIX_SIZE (1 << RADIX_BITS)
#define RADIX_MASK (RADIX_SIZE - 1)
#define RADIX_PASSES (32 / RADIX_BITS)

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t error = call;                                                                  \
        if(error != cudaSuccess) {                                                                 \
            fprintf(stderr,                                                                        \
                    "CUDA error at %s:%d - %s\n",                                                  \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    cudaGetErrorString(error));                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while(0)

namespace cg = cooperative_groups;

// Kernel declaration
__global__ void k_radixSort(unsigned int *keysIn_d,
                            unsigned int *keysOut_d,
                            unsigned int *globalHistograms_d,
                            int numElements);

#endif // RADIX_SORT_H