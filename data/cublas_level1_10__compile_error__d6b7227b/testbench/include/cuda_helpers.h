#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__,   \
              __LINE__, cudaGetErrorString(err));                              \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error in file '%s' in line %i.\n", __FILE__,     \
              __LINE__);                                                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#endif // CUDA_HELPERS_H