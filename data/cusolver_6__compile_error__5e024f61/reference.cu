#include "lu_factorization.h"

void lu_factorization(int m, const std::vector<double> &A,
                      std::vector<double> &LU, std::vector<int> &Ipiv,
                      int &info, bool pivot_on) {
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;

  double *d_A = nullptr;
  int *d_Ipiv = nullptr;
  int *d_info = nullptr;
  double *d_work = nullptr;
  int lwork = 0;

  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv),
                        sizeof(int) * Ipiv.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(),
                             cudaMemcpyHostToDevice, stream));

  CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, m, m, d_A, m, &lwork));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

  if (pivot_on) {
    CUSOLVER_CHECK(
        cusolverDnDgetrf(cusolverH, m, m, d_A, m, d_work, d_Ipiv, d_info));
  } else {
    CUSOLVER_CHECK(
        cusolverDnDgetrf(cusolverH, m, m, d_A, m, d_work, nullptr, d_info));
  }

  CUDA_CHECK(cudaMemcpyAsync(LU.data(), d_A, sizeof(double) * m * m,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int) * Ipiv.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost,
                             stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_Ipiv));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(d_work));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
  CUDA_CHECK(cudaStreamDestroy(stream));
}