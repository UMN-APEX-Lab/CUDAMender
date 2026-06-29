#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define cudaCheckErrors(msg)                                                                 \
    do                                                                                       \
    {                                                                                        \
        cudaError_t __err = cudaGetLastError();                                              \
        if (__err != cudaSuccess)                                                            \
        {                                                                                    \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)", msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__);                                                     \
            fprintf(stderr, "*** FAILED - ABORTING");                                        \
            exit(1);                                                                         \
        }                                                                                    \
    }                                                                                        \
    while (0)

#endif // CUDA_HELPERS_H