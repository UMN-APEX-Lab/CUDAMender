#ifndef DENSE_TO_SPARSE_BLOCKED_ELL_H
#define DENSE_TO_SPARSE_BLOCKED_ELL_H

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iostream>
#include <vector>

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

void dense_to_sparse_blocked_ell(int num_rows, int num_cols, int ell_blk_size,
                                 int ell_width,
                                 const std::vector<float> &h_dense,
                                 const std::vector<int> &h_ell_columns,
                                 std::vector<float> &h_ell_values);

#endif // DENSE_TO_SPARSE_BLOCKED_ELL_H