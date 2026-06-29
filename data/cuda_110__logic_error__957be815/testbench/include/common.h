#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_SIZE (256)
#define CUDA_CHECK(call) \
do { \
       cudaError_t error = call; \
       if (error != cudaSuccess) { \
           fprintf(stderr, "CUDA error at %s:%d %s\n", \
                   __FILE__, __LINE__, \
                   cudaGetErrorString(error)); \
           exit(EXIT_FAILURE); \
       } \
} while(0)

#endif // COMMON_H