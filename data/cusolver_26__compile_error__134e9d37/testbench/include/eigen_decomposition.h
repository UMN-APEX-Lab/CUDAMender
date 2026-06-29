#ifndef EIGEN_DECOMPOSITION_H
#define EIGEN_DECOMPOSITION_H

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <stdexcept>
#include <vector>

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
      fprintf(stderr, "cuSOLVER error in file '%s' in line %i.\n", __FILE__,   \
              __LINE__);                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

void compute_eigen_decomposition(int m, const std::vector<double> &A,
                                 std::vector<double> &W,
                                 std::vector<double> &V);

#endif // EIGEN_DECOMPOSITION_H