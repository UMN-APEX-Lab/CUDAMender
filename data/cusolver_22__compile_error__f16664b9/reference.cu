#include "compute_svd.h"

void compute_svd(int m, int n, std::vector<double> &A, std::vector<double> &S,
                 std::vector<double> &U, std::vector<double> &V,
                 double &h_err_sigma) {
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnParams_t params = NULL;
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

  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

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
  CUDA_CHECK(cudaMemsetAsync(d_U, 0, sizeof(double) * U.size(), stream));
  CUDA_CHECK(cudaMemsetAsync(d_V, 0, sizeof(double) * V.size(), stream));

  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  const int econ = 1;

  CUSOLVER_CHECK(cusolverDnXgesvdp_bufferSize(
      cusolverH, params, jobz, econ, m, n, CUDA_R_64F, d_A, m, CUDA_R_64F, d_S,
      CUDA_R_64F, d_U, m, CUDA_R_64F, d_V, n, CUDA_R_64F,
      &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));
  if (workspaceInBytesOnHost > 0) {
    h_work = malloc(workspaceInBytesOnHost);
    if (h_work == nullptr) {
      throw std::runtime_error("Error: Host workspace allocation failed.");
    }
  }

  CUSOLVER_CHECK(
      cusolverDnXgesvdp(cusolverH, params, jobz, econ, m, n, CUDA_R_64F, d_A, m,
                        CUDA_R_64F, d_S, CUDA_R_64F, d_U, m, CUDA_R_64F, d_V, n,
                        CUDA_R_64F, d_work, workspaceInBytesOnDevice, h_work,
                        workspaceInBytesOnHost, d_info, &h_err_sigma));

  CUDA_CHECK(cudaMemcpyAsync(S.data(), d_S, sizeof(double) * S.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(U.data(), d_U, sizeof(double) * U.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(V.data(), d_V, sizeof(double) * V.size(),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_S));
  CUDA_CHECK(cudaFree(d_U));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_work));
  CUDA_CHECK(cudaFree(d_info));
  free(h_work);

  CUSOLVER_CHECK(cusolverDnDestroyParams(params));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
  CUDA_CHECK(cudaStreamDestroy(stream));
}