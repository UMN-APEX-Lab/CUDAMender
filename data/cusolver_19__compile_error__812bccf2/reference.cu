#include "cusolver_eigen.h"

void compute_eigenvalues_and_vectors(int64_t n,
                                     const std::vector<cuDoubleComplex> &A,
                                     std::vector<cuDoubleComplex> &W,
                                     std::vector<cuDoubleComplex> &VR) {
  cusolverDnHandle_t cusolverH = NULL;
  cusolverDnParams_t params = NULL;
  cudaStream_t stream = NULL;

  cuDoubleComplex *d_A = nullptr;
  cuDoubleComplex *d_W = nullptr;
  cuDoubleComplex *d_VR = nullptr;
  int *d_info = nullptr;
  void *d_work = nullptr;
  void *h_work = nullptr;
  size_t workspaceInBytesOnDevice = 0;
  size_t workspaceInBytesOnHost = 0;

  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  CUDA_CHECK(cudaMalloc(&d_A, sizeof(cuDoubleComplex) * A.size()));
  CUDA_CHECK(cudaMalloc(&d_W, sizeof(cuDoubleComplex) * W.size()));
  CUDA_CHECK(cudaMalloc(&d_VR, sizeof(cuDoubleComplex) * VR.size()));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(cuDoubleComplex) * A.size(),
                             cudaMemcpyHostToDevice, stream));

  CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      cusolverH, params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
      n, CUDA_C_64F, d_A, n, CUDA_C_64F, d_W, CUDA_C_64F, nullptr, 1,
      CUDA_C_64F, d_VR, n, CUDA_C_64F, &workspaceInBytesOnDevice,
      &workspaceInBytesOnHost));

  CUDA_CHECK(cudaMalloc(&d_work, workspaceInBytesOnDevice));
  CUDA_CHECK(cudaMallocHost(&h_work, workspaceInBytesOnHost));

  CUSOLVER_CHECK(cusolverDnXgeev(
      cusolverH, params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
      n, CUDA_C_64F, d_A, n, CUDA_C_64F, d_W, CUDA_C_64F, nullptr, 1,
      CUDA_C_64F, d_VR, n, CUDA_C_64F, d_work, workspaceInBytesOnDevice, h_work,
      workspaceInBytesOnHost, d_info));

  CUDA_CHECK(cudaMemcpyAsync(W.data(), d_W, sizeof(cuDoubleComplex) * W.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(VR.data(), d_VR,
                             sizeof(cuDoubleComplex) * VR.size(),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_W));
  CUDA_CHECK(cudaFree(d_VR));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFree(d_work));
  CUDA_CHECK(cudaFreeHost(h_work));

  CUSOLVER_CHECK(cusolverDnDestroyParams(params));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
  CUDA_CHECK(cudaStreamDestroy(stream));
}