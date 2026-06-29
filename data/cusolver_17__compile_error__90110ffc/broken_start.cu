#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "compute_eigenvalues.h"
#include "cusolver_helpers.h"

void compute_eigenvalues_and_vectors(int64_t n, const std::vector<double> &A,
                                     std::vector<double> &W,
                                     std::vector<double> &VR) {
    W.resize(static_cast<size_t>(2 * n));
    VR.resize(static_cast<size_t>(n * n));

    if (n == 0) {
        return;
    }

    const int64_t lda = n;
    const int64_t ldvr = n;

    double *d_A = nullptr;
    double *d_W = nullptr;
    double *d_VR = nullptr;
    int *d_info = nullptr;
    void *d_work = nullptr;
    void *h_work = nullptr;

    cusolverDnHandle_t handle = nullptr;
    cusolverDnParams_t params = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * static_cast<size_t>(n * n)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_W), sizeof(double) * static_cast<size_t>(2 * n)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_VR), sizeof(double) * static_cast<size_t>(n * n)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(),
                          sizeof(double) * static_cast<size_t>(n * n),
                          cudaMemcpyHostToDevice));

    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    size_t workspace_device_bytes = 0;
    size_t workspace_host_bytes = 0;

    CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
        handle,
        params,
        CUSOLVER_EIG_MODE_NOVECTOR,
        CUSOLVER_EIG_MODE_VECTOR,
        n,
        CUDA_R_64F,
        d_A,
        lda,
        CUDA_R_64F,
        d_W,
        CUDA_R_64F,
        nullptr,
        1,
        CUDA_R_64F,
        d_VR,
        ldvr,
        CUDA_R_64F,
        &workspace_device_bytes,
        &workspace_host_bytes));

    if (workspace_device_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_work, workspace_device_bytes));
    }
    if (workspace_host_bytes > 0) {
        CUDA_CHECK(cudaMallocHost(&h_work, workspace_host_bytes));
    }

    CUSOLVER_CHECK(cusolverDnXgeev(
        handle,
        params,
        CUSOLVER_EIG_MODE_NOVECTOR,
        CUSOLVER_EIG_MODE_VECTOR,
        n,
        CUDA_R_64F,
        d_A,
        lda,
        CUDA_R_64F,
        d_W,
        CUDA_R_64F,
        nullptr,
        1,
        CUDA_R_64F,
        d_VR,
        ldvr,
        CUDA_R_64F,
        d_work,
        workspace_device_bytes,
        h_work,
        workspace_host_bytes,
        d_info));

    int info = 0;
    CUDA_CHECK(cudaMemcpy(W.data(), d_W,
                          sizeof(double) * static_cast<size_t>(2 * n),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(VR.data(), d_VR,
                          sizeof(double) * static_cast<size_t>(n * n),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_work) {
        CUDA_CHECK(cudaFreeHost(h_work));
    }
    if (d_work) {
        CUDA_CHECK(cudaFree(d_work));
    }
    if (d_info) {
        CUDA_CHECK(cudaFree(d_info));
    }
    if (d_VR) {
        CUDA_CHECK(cudaFree(d_VR));
    }
    if (d_W) {
        CUDA_CHECK(cudaFree(d_W));
    }
    if (d_A) {
        CUDA_CHECK(cudaFree(d_A));
    }
    if (params) {
        CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    }
    if (handle) {
        CUSOLVER_CHECK(cusolverDnDestroy(handle));
    }

    if (info != 0) {
        throw std::runtime_error("cusolverDnXgeev failed");
    }
}