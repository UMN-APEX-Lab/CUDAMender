#include "solve_matrix.h"

void solve_matrix(int N, int nrhs, const std::vector<double> &hA,
                  const std::vector<double> &hB, std::vector<double> &hX) {
  cusolverDnHandle_t handle{};
  cusolverDnIRSParams_t gesv_params{};
  cusolverDnIRSInfos_t gesv_info{};

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUSOLVER_CHECK(cusolverDnCreate(&handle));
  CUSOLVER_CHECK(cusolverDnSetStream(handle, stream));

  CUSOLVER_CHECK(cusolverDnIRSParamsCreate(&gesv_params));
  CUSOLVER_CHECK(cusolverDnIRSParamsSetSolverPrecisions(
      gesv_params, CUSOLVER_R_64F, CUSOLVER_R_32F));
  CUSOLVER_CHECK(cusolverDnIRSParamsSetRefinementSolver(
      gesv_params, CUSOLVER_IRS_REFINE_CLASSICAL));
  CUSOLVER_CHECK(cusolverDnIRSInfosCreate(&gesv_info));

  double *dA = nullptr;
  double *dB = nullptr;
  double *dX = nullptr;
  cusolver_int_t *dipiv = nullptr;
  cusolver_int_t *dinfo = nullptr;
  void *dwork = nullptr;
  size_t dwork_size = 0;

  CUDA_CHECK(cudaMalloc(&dA, N * N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dB, N * nrhs * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dX, N * nrhs * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&dipiv, N * sizeof(cusolver_int_t)));
  CUDA_CHECK(cudaMalloc(&dinfo, sizeof(cusolver_int_t)));

  CUDA_CHECK(cudaMemcpyAsync(dA, hA.data(), N * N * sizeof(double),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dB, hB.data(), N * nrhs * sizeof(double),
                             cudaMemcpyHostToDevice, stream));

  CUSOLVER_CHECK(
      cusolverDnIRSXgesv_bufferSize(handle, gesv_params, N, nrhs, &dwork_size));
  CUDA_CHECK(cudaMalloc(&dwork, dwork_size));

  cusolver_int_t iter;
  CUSOLVER_CHECK(cusolverDnIRSXgesv(handle, gesv_params, gesv_info, N, nrhs, dA,
                                    N, dB, N, dX, N, dwork, dwork_size, &iter,
                                    dinfo));

  CUDA_CHECK(cudaMemcpyAsync(hX.data(), dX, N * nrhs * sizeof(double),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dX));
  CUDA_CHECK(cudaFree(dipiv));
  CUDA_CHECK(cudaFree(dinfo));
  CUDA_CHECK(cudaFree(dwork));

  CUSOLVER_CHECK(cusolverDnIRSInfosDestroy(gesv_info));
  CUSOLVER_CHECK(cusolverDnIRSParamsDestroy(gesv_params));
  CUSOLVER_CHECK(cusolverDnDestroy(handle));
  CUDA_CHECK(cudaStreamDestroy(stream));
}