#include "gather.h"
#include "cuda_helpers.h"

void gather(int size, int nnz, const std::vector<int> &hX_indices,
            const std::vector<float> &hY, std::vector<float> &hX_values) {
  int *dX_indices = nullptr;
  float *dY = nullptr, *dX_values = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&dX_indices, nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&dX_values, nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&dY, size * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dX_indices, hX_indices.data(), nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(dY, hY.data(), size * sizeof(float), cudaMemcpyHostToDevice));

  cusparseHandle_t handle = nullptr;
  cusparseSpVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;

  CHECK_CUSPARSE(cusparseCreate(&handle));
  CHECK_CUSPARSE(cusparseCreateSpVec(&vecX, size, nnz, dX_indices, dX_values,
                                     CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, size, dY, CUDA_R_32F));

  CHECK_CUSPARSE(cusparseGather(handle, vecY, vecX));

  CHECK_CUDA(cudaMemcpy(hX_values.data(), dX_values, nnz * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUSPARSE(cusparseDestroySpVec(vecX));
  CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
  CHECK_CUSPARSE(cusparseDestroy(handle));
  CHECK_CUDA(cudaFree(dX_indices));
  CHECK_CUDA(cudaFree(dX_values));
  CHECK_CUDA(cudaFree(dY));
}