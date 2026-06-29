#include "include/cusolver_eigen.h"

#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline void check_cuda(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(status));
    }
}

inline void check_cusolver(cusolverStatus_t status, const char* msg) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": cuSOLVER error code " + std::to_string(static_cast<int>(status)));
    }
}

}  // namespace

void compute_eigenvalues_and_vectors(int64_t n,
                                     const std::vector<cuDoubleComplex>& A,
                                     std::vector<cuDoubleComplex>& W,
                                     std::vector<cuDoubleComplex>& VR) {
    if (n < 0) {
        throw std::invalid_argument("n must be non-negative");
    }
    if (static_cast<int64_t>(A.size()) != n * n) {
        throw std::invalid_argument("A must have size n*n");
    }

    W.resize(static_cast<size_t>(n));
    VR.resize(static_cast<size_t>(n * n));

    if (n == 0) {
        return;
    }

    cusolverDnHandle_t handle = nullptr;
    cusolverDnParams_t params = nullptr;

    cuDoubleComplex* d_A = nullptr;
    cuDoubleComplex* d_W = nullptr;
    cuDoubleComplex* d_VR = nullptr;
    void* d_work = nullptr;
    void* h_work = nullptr;
    int* d_info = nullptr;

    try {
        check_cusolver(cusolverDnCreate(&handle), "cusolverDnCreate failed");
        check_cusolver(cusolverDnCreateParams(&params), "cusolverDnCreateParams failed");

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(cuDoubleComplex) * static_cast<size_t>(n * n)),
                   "cudaMalloc d_A failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_W), sizeof(cuDoubleComplex) * static_cast<size_t>(n)),
                   "cudaMalloc d_W failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_VR), sizeof(cuDoubleComplex) * static_cast<size_t>(n * n)),
                   "cudaMalloc d_VR failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)),
                   "cudaMalloc d_info failed");

        check_cuda(cudaMemcpy(d_A, A.data(), sizeof(cuDoubleComplex) * static_cast<size_t>(n * n),
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy A H2D failed");

        size_t workspace_device_bytes = 0;
        size_t workspace_host_bytes = 0;

        check_cusolver(
            cusolverDnXgeev_bufferSize(
                handle,
                params,
                CUSOLVER_EIG_MODE_NOVECTOR,
                CUSOLVER_EIG_MODE_VECTOR,
                n,
                CUDA_C_64F,
                d_A,
                n,
                CUDA_C_64F,
                d_W,
                CUDA_C_64F,
                nullptr,
                n,
                CUDA_C_64F,
                d_VR,
                n,
                CUDA_C_64F,
                &workspace_device_bytes,
                &workspace_host_bytes),
            "cusolverDnXgeev_bufferSize failed");

        if (workspace_device_bytes > 0) {
            check_cuda(cudaMalloc(&d_work, workspace_device_bytes), "cudaMalloc d_work failed");
        }
        if (workspace_host_bytes > 0) {
            h_work = ::operator new(workspace_host_bytes);
        }

        check_cusolver(
            cusolverDnXgeev(
                handle,
                params,
                CUSOLVER_EIG_MODE_NOVECTOR,
                CUSOLVER_EIG_MODE_VECTOR,
                n,
                CUDA_C_64F,
                d_A,
                n,
                CUDA_C_64F,
                d_W,
                CUDA_C_64F,
                nullptr,
                n,
                CUDA_C_64F,
                d_VR,
                n,
                CUDA_C_64F,
                d_work,
                workspace_device_bytes,
                h_work,
                workspace_host_bytes,
                d_info),
            "cusolverDnXgeev failed");

        int info = 0;
        check_cuda(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost),
                   "cudaMemcpy info D2H failed");
        if (info != 0) {
            throw std::runtime_error("cusolverDnXgeev returned info = " + std::to_string(info));
        }

        check_cuda(cudaMemcpy(W.data(), d_W, sizeof(cuDoubleComplex) * static_cast<size_t>(n),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy W D2H failed");
        check_cuda(cudaMemcpy(VR.data(), d_VR, sizeof(cuDoubleComplex) * static_cast<size_t>(n * n),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy VR D2H failed");

        if (d_info) cudaFree(d_info);
        if (d_work) cudaFree(d_work);
        if (d_VR) cudaFree(d_VR);
        if (d_W) cudaFree(d_W);
        if (d_A) cudaFree(d_A);
        if (h_work) ::operator delete(h_work);
        if (params) cusolverDnDestroyParams(params);
        if (handle) cusolverDnDestroy(handle);
    } catch (...) {
        if (d_info) cudaFree(d_info);
        if (d_work) cudaFree(d_work);
        if (d_VR) cudaFree(d_VR);
        if (d_W) cudaFree(d_W);
        if (d_A) cudaFree(d_A);
        if (h_work) ::operator delete(h_work);
        if (params) cusolverDnDestroyParams(params);
        if (handle) cusolverDnDestroy(handle);
        throw;
    }
}