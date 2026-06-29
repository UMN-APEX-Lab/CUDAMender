#ifndef GRAPH_CONFIG_H
#define GRAPH_CONFIG_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) {                                     \
    cudaError_t error = call;                                  \
    if(error != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

// CUDA settings.
constexpr int BLOCK_SIZE = 256;

// CUDA-graph settings.
constexpr auto HAS_NO_DEPENDENCY = nullptr;
constexpr int ZERO_NODES_AS_DEPENDENCY = 0;
constexpr int ONE_NODE_AS_DEPENDENCY = 1;

// The graph has two nodes. One node for the matrix multiplication, and one device-to-device memcpy node for the in-place feedback loop.
constexpr int NUM_NODES = 2;
constexpr int NODE_KERNEL = 0;
constexpr int NODE_MEMCPY_DEVICE_TO_DEVICE = 1;
constexpr int SELECT_PARAM_MATRIX_SIZE = 0;

// Test settings.
constexpr int MAX_MATRIX_SIZE = 150;
constexpr int MAX_MATRIX_ELEMENTS = MAX_MATRIX_SIZE * MAX_MATRIX_SIZE;

#endif // GRAPH_CONFIG_H