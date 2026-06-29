#ifndef MILLER_RABIN_H
#define MILLER_RABIN_H

#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#define NUMBER_OF_ELEMENTS 16
#define NUMBER_OF_TESTS 9
#define MAX_BLOCKS_PER_SEGMENT 32
#ifndef cudaOccupancyPreferShared
#define cudaOccupancyPreferShared 1
#endif

// Macro for error checking.
#define CUDA_CHECK(call)                                     \
do {                                                         \
    cudaError_t error = call;                                \
    if(error != cudaSuccess) {                               \
        fprintf(stderr,                                      \
            "CUDA Error: %s at %s:%d\n",                     \
                cudaGetErrorString(error),                   \
                    __FILE__,                                \
                    __LINE__);                               \
        exit(error);                                         \
    }                                                        \
} while(0)

extern int gWarpSize;

__global__ void k_millerRabin(unsigned long long* inputNumbers, unsigned long long* primalityResults, int numberOfElements);

__host__ size_t dynamicSMemSizeMillerRabin(int blockSize);

cudaError_t getOptimalLaunchParamsMillerRabin(int numTests, int &optBlockSize, int &blocksPerGrid, float &theoreticalOccupancy);

#endif // MILLER_RABIN_H