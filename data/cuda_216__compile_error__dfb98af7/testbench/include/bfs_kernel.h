#ifndef BFS_KERNEL_H
#define BFS_KERNEL_H

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                         \
    do {                                         \
        cudaError_t err = call;                  \
        if (err != cudaSuccess) {                \
            fprintf(stderr,                      \
                    "CUDA error: %s at %s:%d\n", \
                    cudaGetErrorString(err),     \
                    __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                  \
        }                                        \
    } while (0)

__global__ void k_bfsKernel(
    const int* rowOffsets_d,
    const int* colIndices_d,
    const int* frontier_d,
    int frontierSize,
    int currentLevel,
    int* distances_d,
    int* nextFrontier_d,
    int* nextFrontierSize_d);

#endif // BFS_KERNEL_H