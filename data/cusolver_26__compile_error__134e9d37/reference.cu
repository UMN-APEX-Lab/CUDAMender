#include "eigen_decomposition.h"

void compute_eigen_decomposition(int m, const std::vector<double> &A,
                                 std::vector<double> &W,
                                 std::vector<double> &V) {
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;
  cusolverDnParams_t params = NULL;

  double *d_A = nullptr;
  double *d_W = nullptr;
  int *d_info = nullptr;
  void *d_work = nullptr;
  void *h_work = nullptr;

  size_t workspaceInBytesOnDevice = 0;
  size_t workspaceInBytesOnHost = 0;

  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * W.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(),
                             cudaMemcpyHostToDevice, stream));

  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
      cusolverH, params, jobz, uplo, m, CUDA_R_64F, d_A, m, CUDA_R_64F, d_W,
      CUDA_R_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));
  if (workspaceInBytesOnHost > 0) {
    h_work = malloc(workspaceInBytesOnHost);
    if (h_work == nullptr) {
      throw std::runtime_error("Error: h_work not allocated.");
    }
  }

  CUSOLVER_CHECK(cusolverDnXsyevd(cusolverH, params, jobz, uplo, m, CUDA_R_64F,
                                  d_A, m, CUDA_R_64F, d_W, CUDA_R_64F, d_work,
                                  workspaceInBytesOnDevice, h_work,
                                  workspaceInBytesOnHost, d_info));

  CUDA_CHECK(cudaMemcpyAsync(V.data(), d_A, sizeof(double) * V.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(W.data(), d_W, sizeof(double) * W.size(),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_W));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(d_work));
  free(h_work);

  CUSOLVER_CHECK(cusolverDnDestroyParams(params));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
  CUDA_CHECK(cudaStreamDestroy(stream));
}