#include "trtri.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t _status = (call);                                             \
        if (_status != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +                \
                                     cudaGetErrorString(_status));                \
        }                                                                         \
    } while (0)

#define CHECK_CUSOLVER(call)                                                      \
    do {                                                                          \
        cusolverStatus_t _status = (call);                                        \
        if (_status != CUSOLVER_STATUS_SUCCESS) {                                 \
            throw std::runtime_error("cuSOLVER error");                           \
        }                                                                         \
    } while (0)

void trtri(int n, std::vector<double>& A, cublasFillMode_t uplo,
           cublasDiagType_t diag) {
    if (n < 0) {
        throw std::invalid_argument("n must be non-negative");
    }
    if (static_cast<int>(A.size()) != n * n) {
        throw std::invalid_argument("A must have size n*n");
    }
    if (n == 0) {
        return;
    }

    cusolverDnHandle_t handle = nullptr;
    double* dA = nullptr;
    int* dInfo = nullptr;
    void* dWork = nullptr;
    void* hWork = nullptr;

    try {
        CHECK_CUSOLVER(cusolverDnCreate(&handle));

        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dA), sizeof(double) * A.size()));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dInfo), sizeof(int)));

        CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice));

        // Row-major A in memory is equivalent to column-major A^T.
        // Inverting A^T in column-major yields (A^{-1})^T, which when copied back
        // to the same row-major layout is exactly A^{-1}. Triangular orientation flips.
        cublasFillMode_t transposed_uplo =
            (uplo == CUBLAS_FILL_MODE_UPPER) ? CUBLAS_FILL_MODE_LOWER
                                             : CUBLAS_FILL_MODE_UPPER;

        size_t dWorkSize = 0;
        size_t hWorkSize = 0;

        CHECK_CUSOLVER(cusolverDnXtrtri_bufferSize(
            handle,
            transposed_uplo,
            diag,
            static_cast<int64_t>(n),
            CUDA_R_64F,
            dA,
            static_cast<int64_t>(n),
            CUDA_R_64F,
            &dWorkSize,
            &hWorkSize));

        if (dWorkSize > 0) {
            CHECK_CUDA(cudaMalloc(&dWork, dWorkSize));
        }
        if (hWorkSize > 0) {
            hWork = ::operator new(hWorkSize);
        }

        CHECK_CUSOLVER(cusolverDnXtrtri(
            handle,
            transposed_uplo,
            diag,
            static_cast<int64_t>(n),
            CUDA_R_64F,
            dA,
            static_cast<int64_t>(n),
            CUDA_R_64F,
            dWork,
            dWorkSize,
            hWork,
            hWorkSize,
            dInfo));

        int info = 0;
        CHECK_CUDA(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) {
            throw std::runtime_error("cusolverDnXtrtri failed with info = " + std::to_string(info));
        }

        CHECK_CUDA(cudaMemcpy(A.data(), dA, sizeof(double) * A.size(), cudaMemcpyDeviceToHost));

        if (hWork) {
            ::operator delete(hWork);
        }
        if (dWork) {
            cudaFree(dWork);
        }
        if (dInfo) {
            cudaFree(dInfo);
        }
        if (dA) {
            cudaFree(dA);
        }
        if (handle) {
            cusolverDnDestroy(handle);
        }
    } catch (...) {
        if (hWork) {
            ::operator delete(hWork);
        }
        if (dWork) {
            cudaFree(dWork);
        }
        if (dInfo) {
            cudaFree(dInfo);
        }
        if (dA) {
            cudaFree(dA);
        }
        if (handle) {
            cusolverDnDestroy(handle);
        }
        throw;
    }
}