#include "scale_vector.h"

void scale_vector(int n, double alpha, const std::vector<double> &in,
                  std::vector<double> &out) {
  cublasHandle_t cublasH = nullptr;
  cudaStream_t stream = nullptr;
  double *d_in = nullptr;
  double *d_out = nullptr;

  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(double)));

  CUDA_CHECK(
      cudaMemcpyAsync(d_in, in.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));

  CUDA_CHECK(
      cudaMemcpyAsync(d_out, d_in, n * sizeof(double), cudaMemcpyDeviceToDevice, stream));

  CUBLAS_CHECK(cublasScalEx(cublasH, n, &alpha, CUDA_R_64F, d_out, CUDA_R_64F,
                            1, CUDA_R_64F));

  CUDA_CHECK(cudaMemcpyAsync(out.data(), d_out, n * sizeof(double),
                        cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUBLAS_CHECK(cublasDestroy(cublasH));
  CUDA_CHECK(cudaStreamDestroy(stream));
}