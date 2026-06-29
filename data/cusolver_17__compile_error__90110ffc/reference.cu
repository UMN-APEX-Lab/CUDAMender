#include "compute_eigenvalues.h"
#include "cusolver_helpers.h"

void compute_eigenvalues_and_vectors(int64_t n, const std::vector<double> &A,
                                     std::vector<double> &W,
                                     std::vector<double> &VR) {
  cusolverDnHandle_t cusolverH = NULL;
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;
  cusolverDnParams_t params = NULL;

  double *d_A = nullptr, *d_W = nullptr, *d_VR = nullptr;
  int *d_info = nullptr;
  void *d_work = nullptr, *h_work = nullptr;
  size_t workspaceInBytesOnDevice = 0, workspaceInBytesOnHost = 0;

  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));
  CUSOLVER_CHECK(cusolverDnCreateParams(&params));

  // Allocate device memory with extra padding for cuSOLVER internal operations
  size_t padded_size = (n + 1) * (n + 1);
  CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * padded_size));
  CUDA_CHECK(cudaMalloc(&d_W, sizeof(double) * W.size()));
  CUDA_CHECK(cudaMalloc(&d_VR, sizeof(double) * VR.size()));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  // Initialize all memory to zero before copying data
  CUDA_CHECK(cudaMemset(d_A, 0, sizeof(double) * padded_size));
  CUDA_CHECK(cudaMemset(d_W, 0, sizeof(double) * W.size()));
  CUDA_CHECK(cudaMemset(d_VR, 0, sizeof(double) * VR.size()));
  CUDA_CHECK(cudaMemset(d_info, 0, sizeof(int)));

  // Copy input matrix
  CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(double) * A.size(),
                        cudaMemcpyHostToDevice));

  CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
      cusolverH, params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
      n, CUDA_R_64F, d_A, n, CUDA_R_64F, d_W, CUDA_R_64F, nullptr, 1,
      CUDA_R_64F, d_VR, n, CUDA_R_64F, &workspaceInBytesOnDevice,
      &workspaceInBytesOnHost));

  CUDA_CHECK(cudaMallocHost(&h_work, workspaceInBytesOnHost));
  CUDA_CHECK(cudaMalloc(&d_work, workspaceInBytesOnDevice));

  memset(h_work, 0, workspaceInBytesOnHost);
  CUDA_CHECK(cudaMemset(d_work, 0, workspaceInBytesOnDevice));

  CUSOLVER_CHECK(cusolverDnXgeev(
      cusolverH, params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR,
      n, CUDA_R_64F, d_A, n, CUDA_R_64F, d_W, CUDA_R_64F, nullptr, 1,
      CUDA_R_64F, d_VR, n, CUDA_R_64F, d_work, workspaceInBytesOnDevice, h_work,
      workspaceInBytesOnHost, d_info));

  CUDA_CHECK(cudaMemcpyAsync(W.data(), d_W, sizeof(double) * W.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(VR.data(), d_VR, sizeof(double) * VR.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_W));
  CUDA_CHECK(cudaFree(d_VR));
  CUDA_CHECK(cudaFree(d_info));
  CUDA_CHECK(cudaFreeHost(h_work));
  CUDA_CHECK(cudaFree(d_work));
  CUSOLVER_CHECK(cusolverDnDestroyParams(params));
  CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
  CUBLAS_CHECK(cublasDestroy(cublasH));
  CUDA_CHECK(cudaStreamDestroy(stream));
}