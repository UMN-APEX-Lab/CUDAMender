#ifndef LBP_KERNEL_H
#define LBP_KERNEL_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = call;                                              \
        if(error != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(error), __FILE__, __LINE__);            \
            exit(error);                                                       \
        }                                                                      \
    } while(0)

#define NUM_TEST_CASES 8

#ifndef cudaOccupancyPreferShared
#define cudaOccupancyPreferShared 1
#endif

// Maximum number of thread blocks per image segment for processing
const int MAX_BLOCKS_PER_SEGMENT = 32;

// CUDA kernel for Local Binary Pattern computation
__global__ void k_localBinaryKernel(const unsigned char* input, unsigned char* output, 
                                     int width, int segHeight, int numNeighbors, float radius);

// Computes dynamic shared memory size for given block size
size_t dynamicSMemSizeFunc(int blockSize);

// Determines optimal block and grid sizes using occupancy APIs
cudaError_t getOptimalLaunchParams(int segmentSize, int &optBlockSize, 
                                    int &blocksPerGrid, float &theoreticalOccupancy);

#endif // LBP_KERNEL_H