#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cstdio>

#define CUDA_CHECK(call)                                                           \
do {                                                                               \
        cudaError_t error = call;                                                  \
        if (error != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),\
                    __FILE__, __LINE__);                                           \
            exit(error);                                                           \
        }                                                                          \
} while (0)

// Define the maximum number of threads per block (adjust as needed)
#define MAX_THREADS 1024
#define TOLERANCE   1e-2
#define BLOCKSIZE   4
#define MAXVALUE    9999

#endif // CUDA_COMMON_H