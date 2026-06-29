#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Error-checking macro
#define CUDA_CHECK(call)                                                   \
{                                                                          \
    cudaError_t error = call;                                              \
    if(error != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(error));           \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}

#endif // CUDA_UTILS_H