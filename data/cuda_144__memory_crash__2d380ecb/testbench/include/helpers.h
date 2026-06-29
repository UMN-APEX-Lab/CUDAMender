#ifndef HELPERS_H
#define HELPERS_H

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                           \
do {                                                                               \
        cudaError_t error = call;                                                  \
        if (error != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),\
                    __FILE__, __LINE__);                                           \
            exit(error);                                                           \
        }                                                                          \
} while (0)

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

__device__ __forceinline__ void manual_load_fragment_A(const uint8_t* src, uint32_t frag[2], int stride) {
    uint32_t reg0 = uint32_t(src[0]) | (uint32_t(src[1]) << 8) | (uint32_t(src[2]) << 16) | (uint32_t(src[3]) << 24);
    uint32_t reg1 = uint32_t(src[8*stride]) | (uint32_t(src[8*stride + 1]) << 8) |
                    (uint32_t(src[8*stride + 2]) << 16) | (uint32_t(src[8*stride + 3]) << 24);
    frag[0] = reg0;
    frag[1] = reg1;
}

__device__ __forceinline__ uint32_t manual_load_fragment_B(const uint8_t* src) {
    int lane = threadIdx.x % 32;
    uint8_t v0 = src[0 + lane/4];
    uint8_t v1 = src[8 + lane/4];
    uint8_t v2 = src[16 + lane/4];
    uint8_t v3 = src[24 + lane/4];
    uint32_t frag = (uint32_t)v0 | ((uint32_t)v1 << 8) | ((uint32_t)v2 << 16) | ((uint32_t)v3 << 24);
    return frag;
}

#endif // HELPERS_H