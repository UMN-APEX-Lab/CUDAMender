#ifndef CUSOLVER_HELPERS_H
#define CUSOLVER_HELPERS_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__,   \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUSOLVER_CHECK(call)                                                   \
  {                                                                            \
    cusolverStatus_t status = call;                                            \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "CUSOLVER error in file '%s' in line %i.\n", __FILE__,   \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define CUBLAS_CHECK(call)                                                     \
  {                                                                            \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "CUBLAS error in file '%s' in line %i.\n", __FILE__,     \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#endif // CUSOLVER_HELPERS_H