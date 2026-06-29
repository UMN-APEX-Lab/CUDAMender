#include "compute_svd_batched.h"

void compute_svd_batched(int batchSize, int m, int n,
                         const std::vector<float> &A, std::vector<float> &S,
                         std::vector<float> &U, std::vector<float> &V) {
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;

  const int lda = m;
  const int ldu = m;
  const int ldv = n;
  const int rank = n;
  const long long int strideA = static_cast<long long int>(lda * n);
  const long long int strideS = n;
  const long long int strideU = static_cast<long long int>(ldu * n);
  const long long int strideV = static_cast<long long int>(ldv * n);

  float *d_A = nullptr;
  float *d_S = nullptr;
  float *d_U = nullptr;
  float *d_V = nullptr;
  int *d_info = nullptr;

  int lwork = 0;
  float *d_work = nullptr;

  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                        sizeof(float) * strideA * batchSize));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_S),
                        sizeof(float) * strideS * batchSize));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_U),
                        sizeof(float) * strideU * batchSize));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V),
                        sizeof(float) * strideV * batchSize));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int) * batchSize));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(float) * strideA * batchSize,
                             cudaMemcpyHostToDevice, stream));

  CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched_bufferSize(
      cusolverH, CUSOLVER_EIG_MODE_VECTOR, rank, m, n, d_A, lda, strideA, d_S,
      strideS, d_U, ldu, strideU, d_V, ldv, strideV, &lwork, batchSize));

  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork));

  CUSOLVER_CHECK(cusolverDnSgesvdaStridedBatched(
      cusolverH, CUSOLVER_EIG_MODE_VECTOR, rank, m, n, d_A, lda, strideA, d_S,
      strideS, d_U, ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info,
      nullptr, batchSize));

  CUDA_CHECK(cudaMemcpyAsync(S.data(), d_S, sizeof(float) * strideS * batchSize,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(U.data(), d_U, sizeof(float) * strideU * batchSize,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(V.data(), d_V, sizeof(float) * strideV * batchSize,
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_S));
  CUDA_CHECK(cudaFree(d_U));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(d_work));

  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
  CUDA_CHECK(cudaStreamDestroy(stream));
}