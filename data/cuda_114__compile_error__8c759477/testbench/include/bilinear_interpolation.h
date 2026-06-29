#ifndef BILINEAR_INTERPOLATION_H
#define BILINEAR_INTERPOLATION_H

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>

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

#define BLOCK_SIZE 16

__global__ void k_bilinearInterpolation(float *inputMat, int inputWidth, int inputHeight, 
                                         float *outputMat, int outputWidth, int outputHeight);

#endif // BILINEAR_INTERPOLATION_H