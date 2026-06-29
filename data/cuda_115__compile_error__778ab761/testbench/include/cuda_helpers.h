#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call){                                      \
    cudaError_t error = call;                                  \
    if(error != cudaSuccess){                                  \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

#endif // CUDA_HELPERS_H