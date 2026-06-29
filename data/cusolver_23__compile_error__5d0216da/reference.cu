#include "compute_svd.h"
#include "cuda_utils.h"
#include <stdexcept>

void compute_svd(int64_t m, int64_t n, int64_t rank, int64_t p, int64_t iters,
                 std::vector<double> &A, std::vector<double> &S,
                 std::vector<double> &U, std::vector<double> &V) {
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnParams_t params_gesvdr = NULL;
  cudaStream_t stream = NULL;

  double *d_A = nullptr;
  double *d_S = nullptr;
  double *d_U = nullptr;
  double *d_V = nullptr;
  size_t workspaceInBytesOnDevice = 0;
  size_t workspaceInBytesOnHost = 0;
  void *d_work = nullptr;
  void *h_work = nullptr;
  int *d_info = nullptr;
  int info = 0;

  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
  CUSOLVER_CHECK(cusolverDnCreateParams(&params_gesvdr));

  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_S), sizeof(double) * S.size()));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_U), sizeof(double) * U.size()));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_V), sizeof(double) * V.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(),
                             cudaMemcpyHostToDevice, stream));

  CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
      cusolverH, params_gesvdr, 'S', 'S', m, n, rank, p, iters, CUDA_R_64F, d_A,
      m, CUDA_R_64F, d_S, CUDA_R_64F, d_U, m, CUDA_R_64F, d_V, n, CUDA_R_64F,
      &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));
  if (workspaceInBytesOnHost > 0) {
    h_work = malloc(workspaceInBytesOnHost);
    if (h_work == nullptr) {
      throw std::runtime_error("Error: h_work not allocated.");
    }
  }

  CUSOLVER_CHECK(cusolverDnXgesvdr(
      cusolverH, params_gesvdr, 'S', 'S', m, n, rank, p, iters, CUDA_R_64F, d_A,
      m, CUDA_R_64F, d_S, CUDA_R_64F, d_U, m, CUDA_R_64F, d_V, n, CUDA_R_64F,
      d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost,
      d_info));

  CUDA_CHECK(cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(V.data(), d_V, sizeof(double) * V.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (info < 0) {
    throw std::runtime_error("Error: Invalid parameter in cusolverDnXgesvdr.");
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_S));
  CUDA_CHECK(cudaFree(d_U));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(d_work));
  free(h_work);

  CUSOLVER_CHECK(cusolverDnDestroyParams(params_gesvdr));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
  CUDA_CHECK(cudaStreamDestroy(stream));
}