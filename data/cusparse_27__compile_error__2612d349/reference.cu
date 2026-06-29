#include "spsv_sell.h"

void spsv_sell_example(int A_num_rows, int A_num_cols, int A_nnz,
                       const std::vector<int> &hA_sliceOffsets,
                       const std::vector<int> &hA_columns,
                       const std::vector<float> &hA_values,
                       const std::vector<float> &hX, std::vector<float> &hY,
                       float alpha) {
  const int A_slice_size = 2;
  const int A_values_size = hA_values.size();
  const int A_num_slices = (A_num_rows + A_slice_size - 1) / A_slice_size;

  // Device memory management
  int *dA_sliceOffsets, *dA_columns;
  float *dA_values, *dX, *dY;

  CHECK_CUDA(
      cudaMalloc((void **)&dA_sliceOffsets, (A_num_slices + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_values_size * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&dA_values, A_values_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&dX, A_num_cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&dY, A_num_rows * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dA_sliceOffsets, hA_sliceOffsets.data(),
                        (A_num_slices + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns.data(),
                        A_values_size * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dA_values, hA_values.data(),
                        A_values_size * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dX, hX.data(), A_num_cols * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dY, hY.data(), A_num_rows * sizeof(float),
                        cudaMemcpyHostToDevice));

  // CUSPARSE APIs
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnVecDescr_t vecX, vecY;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  cusparseSpSVDescr_t spsvDescr;

  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Create sparse matrix A in SELL format
  CHECK_CUSPARSE(cusparseCreateSlicedEll(
      &matA, A_num_rows, A_num_cols, A_nnz, A_values_size, A_slice_size,
      dA_sliceOffsets, dA_columns, dA_values, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  // Create dense vector X
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F));

  // Create dense vector Y
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F));

  // Create opaque data structure, that holds analysis data between calls.
  CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescr));

  // Specify Lower|Upper fill mode.
  cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE,
                                           &fillmode, sizeof(fillmode)));

  // Specify Unit|Non-Unit diagonal type.
  cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
  CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                           &diagtype, sizeof(diagtype)));

  // allocate an external buffer for analysis
  CHECK_CUSPARSE(cusparseSpSV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, vecY,
      CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufferSize));
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

  CHECK_CUSPARSE(cusparseSpSV_analysis(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, vecY,
      CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer));

  // execute SpSV
  CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, vecY, CUDA_R_32F,
                                    CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr));

  // Copy results back to host
  CHECK_CUDA(cudaMemcpy(hY.data(), dY, A_num_rows * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroySpMat(matA));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
  CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescr));
  CHECK_CUSPARSE(cusparseDestroy(handle));

  // device memory deallocation
  CHECK_CUDA(cudaFree(dBuffer));
  CHECK_CUDA(cudaFree(dA_sliceOffsets));
  CHECK_CUDA(cudaFree(dA_columns));
  CHECK_CUDA(cudaFree(dA_values));
  CHECK_CUDA(cudaFree(dX));
  CHECK_CUDA(cudaFree(dY));
}