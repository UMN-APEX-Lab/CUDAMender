#ifndef HISTOGRAM_HELPERS_H
#define HISTOGRAM_HELPERS_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define NUM_CHANNELS        (4)
#define NUM_ACTIVE_CHANNELS (3)
#define LOWER_LEVEL         (0)
#define UPPER_LEVEL         (256)
#define MAX_NUM_LEVELS      (12)
#define MAX_NUM_BINS        (MAX_NUM_LEVELS - 1)

#define CUDA_CHECK(call)                              \
do {                                                  \
       cudaError_t error = call;                      \
       if (error != cudaSuccess) {                    \
           fprintf(stderr, "CUDA error at %s:%d %s\n",\
                   __FILE__, __LINE__,                \
                   cudaGetErrorString(error));        \
           exit(EXIT_FAILURE);                        \
       }                                              \
} while(0)

#define CUB_CHECK(call)                                                 \
do {                                                                    \
    call;                                                               \
    cudaError_t err = cudaGetLastError();                               \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                cudaGetErrorString(err));                               \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

#endif // HISTOGRAM_HELPERS_H