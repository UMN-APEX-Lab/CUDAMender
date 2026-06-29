#ifndef LBP_H
#define LBP_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Bit positions for 8-neighbor LBP pattern (clockwise from top-left)
#define BIT_TOP_LEFT  7
#define BIT_TOP 6
#define BIT_TOP_RIGHT 5
#define BIT_RIGHT 4
#define BIT_BOTTOM_RIGHT 3
#define BIT_BOTTOM 2
#define BIT_BOTTOM_LEFT 1
#define BIT_LEFT 0

// Halo padding for shared memory (2 * HALO_SIZE for both sides)
#define HALO_PADDING 2

// CUDA error checking macro
#define CUDA_CHECK(call)                                  \
    do {                                                  \
        cudaError_t err = call;                           \
        if (err != cudaSuccess) {                         \
            fprintf(stderr,                               \
                    "CUDA error: %s at %s:%d\n",          \
                    cudaGetErrorString(err),              \
                    __FILE__, __LINE__);                  \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    } while (0)

// Kernel declaration
__global__ void k_computeLBP(const unsigned char* input_d, unsigned char* output_d, int width, int height);

#endif // LBP_H