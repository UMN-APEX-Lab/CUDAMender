#include "dense_to_sparse.h"
#include "cusparse_helpers.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>

void dense_to_sparse_blocked_ell(int num_rows, int num_cols, int ell_blk_size,
                                 int ell_width,
                                 const std::vector<float> &h_dense,
                                 const std::vector<int> &h_ell_columns,
                                 std::vector<float> &h_ell_values) {
  int dense_size = num_rows * num_cols;
  int nnz = ell_width * num_rows;

  float *d_dense = nullptr;
  float *d_ell_values = nullptr;
  int *d_ell_columns = nullptr;
  void *dBuffer = nullptr;
  size_t bufferSize = 0;

  CHECK_CUDA(cudaMalloc((void **)&d_dense, dense_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_ell_columns,
                        nnz / (ell_blk_size * ell_blk_size) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&d_ell_values, nnz * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_dense, h_dense.data(), dense_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_ell_columns, h_ell_columns.data(),
                        nnz / (ell_blk_size * ell_blk_size) * sizeof(int),
                        cudaMemcpyHostToDevice));

  cusparseHandle_t handle;
  cusparseSpMatDescr_t matB;
  cusparseDnMatDescr_t matA;

  CHECK_CUSPARSE(cusparseCreate(&handle));
  CHECK_CUSPARSE(cusparseCreateDnMat(&matA, num_rows, num_cols, num_cols,
                                     d_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW));
  CHECK_CUSPARSE(cusparseCreateBlockedEll(
      &matB, num_rows, num_cols, ell_blk_size, ell_width, d_ell_columns,
      d_ell_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
      handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  CHECK_CUSPARSE(cusparseDenseToSparse_analysis(
      handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));
  CHECK_CUSPARSE(cusparseDenseToSparse_convert(
      handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

  CHECK_CUDA(cudaMemcpy(h_ell_values.data(), d_ell_values, nnz * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUSPARSE(cusparseDestroyDnMat(matA));
  CHECK_CUSPARSE(cusparseDestroySpMat(matB));
  CHECK_CUSPARSE(cusparseDestroy(handle));

  CHECK_CUDA(cudaFree(dBuffer));
  CHECK_CUDA(cudaFree(d_ell_columns));
  CHECK_CUDA(cudaFree(d_ell_values));
  CHECK_CUDA(cudaFree(d_dense));
}