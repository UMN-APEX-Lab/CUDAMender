#ifndef LU_FACTORIZATION_H
#define LU_FACTORIZATION_H

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>

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

// Utility function declaration
void printMatrix(int m, int n, const double *A, int lda, const char *name);

// Main contract function
void lu_factorization(int m, const std::vector<double> &A,
                      std::vector<double> &LU, std::vector<int> &Ipiv,
                      int &info, bool pivot_on);

#endif // LU_FACTORIZATION_H