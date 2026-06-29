#include "sddmm_bsr.h"
#include "cuda_helpers.h"

void sddmm_bsr_example(int A_num_rows, int A_num_cols, int B_num_rows,
                       int B_num_cols, int row_block_dim, int col_block_dim,
                       const std::vector<float> &hA,
                       const std::vector<float> &hB,
                       const std::vector<int> &hC_boffsets,
                       const std::vector<int> &hC_bcolumns,
                       std::vector<float> &hC_values) {
  const int C_num_brows = A_num_rows / row_block_dim;
  const int C_num_bcols = B_num_cols / col_block_dim;
  const int C_bnnz = hC_bcolumns.size();
  const int C_nnz = C_bnnz * row_block_dim * col_block_dim;
  const int lda = A_num_rows;
  const int ldb = B_num_cols;
  const int A_size = lda * A_num_cols;
  const int B_size = ldb * B_num_rows;

  float alpha = 1.0f;
  float beta = 0.0f;

  int *dC_boffsets, *dC_bcolumns;
  float *dC_values, *dB, *dA;
  CHECK_CUDA(cudaMalloc((void **)&dA, A_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&dB, B_size * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc((void **)&dC_boffsets, (C_num_brows + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&dC_bcolumns, C_bnnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&dC_values, C_nnz * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), A_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), B_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC_boffsets, hC_boffsets.data(),
                        (C_num_brows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC_bcolumns, hC_bcolumns.data(), C_bnnz * sizeof(int),
                        cudaMemcpyHostToDevice));

  cusparseHandle_t handle = NULL;
  cusparseDnMatDescr_t matA, matB;
  cusparseSpMatDescr_t matC;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreate(&handle));
  CHECK_CUSPARSE(cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, lda, dA,
                                     CUDA_R_32F, CUSPARSE_ORDER_ROW));
  CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                     CUDA_R_32F, CUSPARSE_ORDER_ROW));
  CHECK_CUSPARSE(cusparseCreateBsr(&matC, C_num_brows, C_num_bcols, C_bnnz,
                                   row_block_dim, col_block_dim, dC_boffsets,
                                   dC_bcolumns, dC_values, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_32F, CUSPARSE_ORDER_ROW));
  CHECK_CUSPARSE(cusparseSDDMM_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  CHECK_CUSPARSE(cusparseSDDMM_preprocess(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
      CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer));
  CHECK_CUSPARSE(cusparseSDDMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                               matB, &beta, matC, CUDA_R_32F,
                               CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer));

  CHECK_CUDA(cudaMemcpy(hC_values.data(), dC_values, C_nnz * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUSPARSE(cusparseDestroyDnMat(matA));
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
  CHECK_CUSPARSE(cusparseDestroySpMat(matC));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  CHECK_CUDA(cudaFree(dBuffer));
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC_boffsets));
  CHECK_CUDA(cudaFree(dC_bcolumns));
  CHECK_CUDA(cudaFree(dC_values));
}