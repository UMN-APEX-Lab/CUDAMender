#ifndef CUSOLVER_HELPERS_H
#define CUSOLVER_HELPERS_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>

// Define CUSOLVER_CHECK macro
#define CUSOLVER_CHECK(call)                                                   \
  {                                                                            \
    cusolverStatus_t err = call;                                               \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "CUSOLVER error: %d at %s:%d\n", err, __FILE__,     \
                   __LINE__);                                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

// Define CUDA_CHECK macro
#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error: %s at %s:%d\n",                        \
                   cudaGetErrorString(err), __FILE__, __LINE__);               \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#endif // CUSOLVER_HELPERS_H