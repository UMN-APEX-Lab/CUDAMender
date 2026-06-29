#ifndef LEAST_SQUARES_COMMON_H
#define LEAST_SQUARES_COMMON_H

#include <cstdio>
#include <cuda_runtime.h>

constexpr int SET_TO_ZERO = 0;
constexpr float EPSILON = 1e-3f;

#define CUDA_CHECK(call)                                        \
do {                                                            \
        cudaError_t error = call;                               \
        if (error != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",       \
                    __FILE__, __LINE__,                         \
                    cudaGetErrorString(error));                 \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
} while(0)

extern const int FIRST_STREAM_INDEX;
extern const int SECOND_STREAM_INDEX;
extern const int OUTPUT_SIZE;

#endif // LEAST_SQUARES_COMMON_H