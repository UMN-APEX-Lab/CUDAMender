#include "include/cublas_rotm.h"
#include "include/cuda_helpers.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <vector>

#if !defined(CHECK_CUDA)
  #if defined(CUDA_CHECK)
    #define CHECK_CUDA CUDA_CHECK
  #endif
#endif

#if !defined(CHECK_CUBLAS)
  #if defined(CUBLAS_CHECK)
    #define CHECK_CUBLAS CUBLAS_CHECK
  #endif
#endif

void cublas_rotm_example(int n, std::vector<double> &A, std::vector<double> &B,
                         const std::vector<double> &param) {
    if (n <= 0) return;

    if (A.size() < static_cast<size_t>(n)) A.resize(n);
    if (B.size() < static_cast<size_t>(n)) B.resize(n);

    double *dA = nullptr;
    double *dB = nullptr;
    cublasHandle_t handle = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dA), sizeof(double) * n));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dB), sizeof(double) * n));

    CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(double) * n, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeof(double) * n, cudaMemcpyHostToDevice));

    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    double hparam[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    const int copy_count = param.size() < 5 ? static_cast<int>(param.size()) : 5;
    for (int i = 0; i < copy_count; ++i) {
        hparam[i] = param[i];
    }

    CHECK_CUBLAS(cublasDrotm(handle, n, dA, 1, dB, 1, hparam));

    CHECK_CUDA(cudaMemcpy(A.data(), dA, sizeof(double) * n, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(B.data(), dB, sizeof(double) * n, cudaMemcpyDeviceToHost));

    if (handle) {
        CHECK_CUBLAS(cublasDestroy(handle));
    }
    if (dA) {
        CHECK_CUDA(cudaFree(dA));
    }
    if (dB) {
        CHECK_CUDA(cudaFree(dB));
    }
}