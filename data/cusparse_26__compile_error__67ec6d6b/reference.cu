#include "spsv_csr.h"
#include "cusparse_utils.h"

void spsv_csr_example(int A_num_rows, int A_num_cols, int A_nnz,
                      const std::vector<int> &hA_csrOffsets,
                      const std::vector<int> &hA_columns,
                      const std::vector<float> &hA_values,
                      const std::vector<float> &hX, std::vector<float> &hY) {
  float alpha = 1.0f;

  int *dA_csrOffsets, *dA_columns;
  float *dA_values, *dX, *dY;
  CHECK_CUDA(
      cudaMalloc((void **)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)))
  CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)))
  CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void **)&dX, A_num_cols * sizeof(float)))
  CHECK_CUDA(cudaMalloc((void **)&dY, A_num_rows * sizeof(float)))

  CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets.data(),
                        (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns.data(), A_nnz * sizeof(int),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dA_values, hA_values.data(), A_nnz * sizeof(float),
                        cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dX, hX.data(), A_num_cols * sizeof(float),
                        cudaMemcpyHostToDevice))

  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  cusparseSpSVDescr_t spsvDescr;
  CHECK_CUSPARSE(cusparseCreate(&handle))
  CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                   dA_csrOffsets, dA_columns, dA_values,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F))
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F))
  CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescr))

  cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE,
                                           &fillmode, sizeof(fillmode)))
  cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                           &diagtype, sizeof(diagtype)))

  CHECK_CUSPARSE(cusparseSpSV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, vecY,
      CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufferSize))
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
  CHECK_CUSPARSE(cusparseSpSV_analysis(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, vecY,
      CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer))
  CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, vecY, CUDA_R_32F,
                                    CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr))

  CHECK_CUDA(cudaMemcpy(hY.data(), dY, A_num_rows * sizeof(float),
                        cudaMemcpyDeviceToHost))

  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
  CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescr));
  CHECK_CUSPARSE(cusparseDestroy(handle))

  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUDA(cudaFree(dA_csrOffsets))
  CHECK_CUDA(cudaFree(dA_columns))
  CHECK_CUDA(cudaFree(dA_values))
  CHECK_CUDA(cudaFree(dX))
  CHECK_CUDA(cudaFree(dY))
}