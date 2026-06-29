#ifndef SCALE_VECTOR_H
#define SCALE_VECTOR_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__     \
                << ": " << cudaGetErrorString(err) << std::endl;               \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "cuBLAS error in " << __FILE__ << " at line " << __LINE__   \
                << ": " << status << std::endl;                                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

void scale_vector(int n, double alpha, const std::vector<double> &in,
                  std::vector<double> &out);

#endif // SCALE_VECTOR_H