#include "gpsv_interleaved_batch.h"
#include "error_checks.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

void solveGpsvInterleavedBatch(
    int n, int batchSize, const std::vector<float> &h_S,
    const std::vector<float> &h_L, const std::vector<float> &h_M,
    const std::vector<float> &h_U, const std::vector<float> &h_W,
    const std::vector<float> &h_B, std::vector<float> &h_X) {
  int full_size = n * batchSize;

  float *d_S0 = nullptr, *d_L0 = nullptr, *d_M0 = nullptr, *d_U0 = nullptr,
        *d_W0 = nullptr;
  float *d_S = nullptr, *d_L = nullptr, *d_M = nullptr, *d_U = nullptr,
        *d_W = nullptr;
  float *d_B = nullptr, *d_X = nullptr;

  CHECK_CUDA(cudaMalloc((void **)&d_S0, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_L0, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_M0, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_U0, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_W0, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_S, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_L, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_M, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_U, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_W, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_B, full_size * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&d_X, full_size * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_S0, h_S.data(), full_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_L0, h_L.data(), full_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_M0, h_M.data(), full_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_U0, h_U.data(), full_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_W0, h_W.data(), full_size * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), full_size * sizeof(float),
                        cudaMemcpyHostToDevice));

  cusparseHandle_t cusparseHandle = NULL;
  cublasHandle_t cublasHandle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
  CHECK_CUBLAS(cublasCreate(&cublasHandle));

  float h_one = 1.0f, h_zero = 0.0f;

  CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, batchSize, n,
                           &h_one, d_S0, n, &h_zero, NULL, n, d_S, batchSize));
  CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, batchSize, n,
                           &h_one, d_L0, n, &h_zero, NULL, n, d_L, batchSize));
  CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, batchSize, n,
                           &h_one, d_M0, n, &h_zero, NULL, n, d_M, batchSize));
  CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, batchSize, n,
                           &h_one, d_U0, n, &h_zero, NULL, n, d_U, batchSize));
  CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, batchSize, n,
                           &h_one, d_W0, n, &h_zero, NULL, n, d_W, batchSize));
  CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, batchSize, n,
                           &h_one, d_B, n, &h_zero, NULL, n, d_X, batchSize));

  size_t bufferSize;
  void *d_buffer = nullptr;
  int algo = 0;

  CHECK_CUSPARSE(cusparseSgpsvInterleavedBatch_bufferSizeExt(
      cusparseHandle, algo, n, d_S, d_L, d_M, d_U, d_W, d_X, batchSize,
      &bufferSize));
  CHECK_CUDA(cudaMalloc((void **)&d_buffer, bufferSize));

  CHECK_CUSPARSE(cusparseSgpsvInterleavedBatch(cusparseHandle, algo, n, d_S,
                                               d_L, d_M, d_U, d_W, d_X,
                                               batchSize, d_buffer));

  CHECK_CUBLAS(cublasSgeam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T, n, batchSize,
                           &h_one, d_X, batchSize, &h_zero, NULL, batchSize, d_B, n));

  CHECK_CUDA(cudaMemcpy(h_X.data(), d_B, full_size * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
  CHECK_CUBLAS(cublasDestroy(cublasHandle));

  CHECK_CUDA(cudaFree(d_S0));
  CHECK_CUDA(cudaFree(d_L0));
  CHECK_CUDA(cudaFree(d_M0));
  CHECK_CUDA(cudaFree(d_U0));
  CHECK_CUDA(cudaFree(d_W0));
  CHECK_CUDA(cudaFree(d_S));
  CHECK_CUDA(cudaFree(d_L));
  CHECK_CUDA(cudaFree(d_M));
  CHECK_CUDA(cudaFree(d_U));
  CHECK_CUDA(cudaFree(d_W));
  CHECK_CUDA(cudaFree(d_B));
  CHECK_CUDA(cudaFree(d_X));
  CHECK_CUDA(cudaFree(d_buffer));
}