#ifndef COMPUTE_SVD_H
#define COMPUTE_SVD_H

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <vector>

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

#define CUSOLVER_CHECK(call)                                                   \
  do {                                                                         \
    cusolverStatus_t status = call;                                            \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "cuSOLVER error in file '%s' in line %i.\n", __FILE__,   \
              __LINE__);                                                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

void compute_svd(int m, int n, std::vector<double> &A, std::vector<double> &S,
                 std::vector<double> &U, std::vector<double> &V,
                 double &h_err_sigma);

#endif // COMPUTE_SVD_H