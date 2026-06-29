#include "trtri.h"
#include "cuda_helpers.h"
#include <stdexcept>

void trtri(int n, std::vector<double> &A, cublasFillMode_t uplo,
           cublasDiagType_t diag) {
  cusolverDnHandle_t handle = NULL;
  cudaStream_t stream = NULL;

  double *d_A = nullptr;
  int *d_info = nullptr;
  int h_info = 0;
  void *d_work = nullptr;
  size_t workspaceInBytesOnDevice = 0;
  void *h_work = nullptr;
  size_t workspaceInBytesOnHost = 0;
  const int lda = n;

  CUSOLVER_CHECK(cusolverDnCreate(&handle));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

  CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * A.size()));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(),
                             cudaMemcpyHostToDevice, stream));

  CUSOLVER_CHECK(cusolverDnXtrtri_bufferSize(
      handle, uplo, diag, n, CUDA_R_64F, d_A, lda, &workspaceInBytesOnDevice,
      &workspaceInBytesOnHost));

  CUDA_CHECK(cudaMalloc(&d_work, workspaceInBytesOnDevice));
  if (workspaceInBytesOnHost > 0) {
    h_work = malloc(workspaceInBytesOnHost);
    if (h_work == nullptr) {
      throw std::runtime_error("Host memory allocation failed.");
    }
  }

  CUSOLVER_CHECK(cusolverDnXtrtri(handle, uplo, diag, n, CUDA_R_64F, d_A, lda,
                                  d_work, workspaceInBytesOnDevice, h_work,
                                  workspaceInBytesOnHost, d_info));

  CUDA_CHECK(cudaMemcpyAsync(&h_info, d_info, sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(A.data(), d_A, sizeof(double) * A.size(),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (h_info != 0) {
    throw std::runtime_error("Matrix inversion failed with info = " +
                             std::to_string(h_info));
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(d_work));
  free(h_work);

  CUSOLVER_CHECK(cusolverDnDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}