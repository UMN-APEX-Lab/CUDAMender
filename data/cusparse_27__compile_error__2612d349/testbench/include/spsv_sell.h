#ifndef SPSV_SELL_H
#define SPSV_SELL_H

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

void spsv_sell_example(int A_num_rows, int A_num_cols, int A_nnz,
                       const std::vector<int> &hA_sliceOffsets,
                       const std::vector<int> &hA_columns,
                       const std::vector<float> &hA_values,
                       const std::vector<float> &hX, std::vector<float> &hY,
                       float alpha);

#endif // SPSV_SELL_H