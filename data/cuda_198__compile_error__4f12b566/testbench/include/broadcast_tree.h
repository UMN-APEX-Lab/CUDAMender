#ifndef BROADCAST_TREE_H
#define BROADCAST_TREE_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <stdio.h>

#define CUDA_CHECK(call) {                                     \
    cudaError_t error = call;                                  \
    if(error != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

// Test-related constants.
constexpr uint32_t MAXIMUM_NUMBER_OF_WARPS_TO_COMMUNICATE = 500;
constexpr uint32_t NUMBER_OF_TESTS = 7;

// Algorithm-related constants.
constexpr uint32_t BITS_PER_BYTE = 8;
constexpr uint32_t BITS_PER_MESSAGE = sizeof(uint32_t) * BITS_PER_BYTE;
constexpr uint32_t DATA_BITS = 24;
constexpr uint32_t ID_BITS = BITS_PER_MESSAGE - DATA_BITS;
constexpr uint32_t ID_MASK = ((1 << ID_BITS) - 1);

// Function to check if a message with a specific ID is received, and then re-send it to two other warps only once and return the decoded data.
__device__ __forceinline__ uint32_t d_broadcast(uint32_t messageId, 
                                                int warpLane, 
                                                uint32_t * message_d, 
                                                int numCommunicatingWarps, 
                                                int globalWarpId);

// Kernel function that sends a sample data from first warp to other warps in the grid without synchronization and with minimal contention.
__global__ void k_broadcastWithHierarchicalPath(uint32_t input, 
                                                uint32_t * message_d, 
                                                int numCommunicatingWarps, 
                                                int * output_d);

#endif // BROADCAST_TREE_H