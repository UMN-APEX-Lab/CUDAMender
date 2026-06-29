#ifndef SOLVE_MATRIX_H
#define SOLVE_MATRIX_H

#include <cassert>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;     \
      std::exit(1);                                                            \
    }                                                                          \
  }

#define CUSOLVER_CHECK(call)                                                   \
  {                                                                            \
    cusolverStatus_t status = call;                                            \
    if (status != CUSOLVER_STATUS_SUCCESS) {                                   \
      std::cerr << "cuSOLVER error: " << status << std::endl;                  \
      std::exit(1);                                                            \
    }                                                                          \
  }

void solve_matrix(int N, int nrhs, const std::vector<double> &hA,
                  const std::vector<double> &hB, std::vector<double> &hX);

#endif // SOLVE_MATRIX_H