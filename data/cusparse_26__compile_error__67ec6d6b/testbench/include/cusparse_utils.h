#ifndef CUSPARSE_UTILS_H
#define CUSPARSE_UTILS_H

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iostream>

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      std::cout << "CUDA API failed at line " << __LINE__                      \
                << " with error: " << cudaGetErrorString(status) << " ("       \
                << status << ")" << std::endl;                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      std::cout << "CUSPARSE API failed at line " << __LINE__                  \
                << " with error: " << cusparseGetErrorString(status) << " ("   \
                << status << ")" << std::endl;                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#endif // CUSPARSE_UTILS_H