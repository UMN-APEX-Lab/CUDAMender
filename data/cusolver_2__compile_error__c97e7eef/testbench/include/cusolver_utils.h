#ifndef CUSOLVER_UTILS_H
#define CUSOLVER_UTILS_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__,  \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUSOLVER_CHECK(call)                                                   \
  {                                                                            \
    cusolverStatus_t err = call;                                               \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "CUSOLVER error in file '%s' in line %i.\n", __FILE__,   \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#endif // CUSOLVER_UTILS_H