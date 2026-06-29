#include "cublas_rotm.h"
#include "cuda_helpers.h"

void cublas_rotm_example(int n, std::vector<double> &A, std::vector<double> &B,
                         const std::vector<double> &param) {
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  double *d_A = nullptr;
  double *d_B = nullptr;

  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * n));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), sizeof(double) * n));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * n,
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(double) * n,
                             cudaMemcpyHostToDevice, stream));

  CUBLAS_CHECK(cublasDrotm(cublasH, n, d_A, 1, d_B, 1, param.data()));

  CUDA_CHECK(cudaMemcpyAsync(A.data(), d_A, sizeof(double) * n,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(double) * n,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));

  CUBLAS_CHECK(cublasDestroy(cublasH));
  CUDA_CHECK(cudaStreamDestroy(stream));
}