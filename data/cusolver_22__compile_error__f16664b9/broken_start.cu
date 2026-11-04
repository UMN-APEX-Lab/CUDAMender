#include "include/compute_svd.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <vector>
#include <stdexcept>
#include <algorithm>

void compute_svd(int m, int n, std::vector<double> &A, std::vector<double> &S,
                 std::vector<double> &U, std::vector<double> &V,
                 double &h_err_sigma) {
    if (m < 0 || n < 0) {
        throw std::invalid_argument("m and n must be non-negative");
    }

    const int64_t lda = (m > 0) ? m : 1;
    const int64_t ldu = (m > 0) ? m : 1;
    const int64_t ldv = (n > 0) ? n : 1;
    const int64_t mn = std::min(m, n);

    if (static_cast<int64_t>(A.size()) != static_cast<int64_t>(m) * static_cast<int64_t>(n)) {
        throw std::invalid_argument("A.size() must be m * n");
    }

    S.resize(static_cast<size_t>(mn));
    U.resize(static_cast<size_t>(ldu) * static_cast<size_t>(m));
    V.resize(static_cast<size_t>(ldv) * static_cast<size_t>(n));

    cusolverDnHandle_t handle = nullptr;
    cusolverDnParams_t params = nullptr;

    CHECK_CUSOLVER(cusolverDnCreate(&handle));
    CHECK_CUSOLVER(cusolverDnCreateParams(&params));

    double *d_A = nullptr;
    double *d_S = nullptr;
    double *d_U = nullptr;
    double *d_V = nullptr;
    int *d_info = nullptr;
    void *d_work = nullptr;
    void *h_work = nullptr;

    const size_t bytesA = static_cast<size_t>(m) * static_cast<size_t>(n) * sizeof(double);
    const size_t bytesS = static_cast<size_t>(mn) * sizeof(double);
    const size_t bytesU = static_cast<size_t>(ldu) * static_cast<size_t>(m) * sizeof(double);
    const size_t bytesV = static_cast<size_t>(ldv) * static_cast<size_t>(n) * sizeof(double);

    if (bytesA > 0) CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    if (bytesS > 0) CHECK_CUDA(cudaMalloc(&d_S, bytesS));
    if (bytesU > 0) CHECK_CUDA(cudaMalloc(&d_U, bytesU));
    if (bytesV > 0) CHECK_CUDA(cudaMalloc(&d_V, bytesV));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

    if (bytesA > 0) {
        CHECK_CUDA(cudaMemcpy(d_A, A.data(), bytesA, cudaMemcpyHostToDevice));
    }

    int64_t workspaceInBytesOnDevice = 0;
    int64_t workspaceInBytesOnHost = 0;

    const signed char jobz = 'V';
    const int econ = 0;

    CHECK_CUSOLVER(cusolverDnXgesvdp_bufferSize(
        handle,
        params,
        jobz,
        econ,
        static_cast<int64_t>(m),
        static_cast<int64_t>(n),
        CUDA_R_64F,
        d_A,
        lda,
        CUDA_R_64F,
        d_S,
        CUDA_R_64F,
        d_U,
        ldu,
        CUDA_R_64F,
        d_V,
        ldv,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    if (workspaceInBytesOnDevice > 0) {
        CHECK_CUDA(cudaMalloc(&d_work, static_cast<size_t>(workspaceInBytesOnDevice)));
    }
    if (workspaceInBytesOnHost > 0) {
        h_work = ::operator new(static_cast<size_t>(workspaceInBytesOnHost));
    }

    CHECK_CUSOLVER(cusolverDnXgesvdp(
        handle,
        params,
        jobz,
        econ,
        static_cast<int64_t>(m),
        static_cast<int64_t>(n),
        CUDA_R_64F,
        d_A,
        lda,
        CUDA_R_64F,
        d_S,
        CUDA_R_64F,
        d_U,
        ldu,
        CUDA_R_64F,
        d_V,
        ldv,
        CUDA_R_64F,
        d_work,
        workspaceInBytesOnDevice,
        h_work,
        workspaceInBytesOnHost,
        d_info,
        &h_err_sigma));

    CHECK_CUDA(cudaDeviceSynchronize());

    int h_info = 0;
    CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        if (h_work) {
            ::operator delete(h_work);
        }
        if (d_work) cudaFree(d_work);
        if (d_info) cudaFree(d_info);
        if (d_V) cudaFree(d_V);
        if (d_U) cudaFree(d_U);
        if (d_S) cudaFree(d_S);
        if (d_A) cudaFree(d_A);
        if (params) cusolverDnDestroyParams(params);
        if (handle) cusolverDnDestroy(handle);
        throw std::runtime_error("cusolverDnXgesvdp failed");
    }

    if (bytesA > 0) CHECK_CUDA(cudaMemcpy(A.data(), d_A, bytesA, cudaMemcpyDeviceToHost));
    if (bytesS > 0) CHECK_CUDA(cudaMemcpy(S.data(), d_S, bytesS, cudaMemcpyDeviceToHost));
    if (bytesU > 0) CHECK_CUDA(cudaMemcpy(U.data(), d_U, bytesU, cudaMemcpyDeviceToHost));
    if (bytesV > 0) CHECK_CUDA(cudaMemcpy(V.data(), d_V, bytesV, cudaMemcpyDeviceToHost));

    if (h_work) ::operator delete(h_work);
    if (d_work) CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(d_info));
    if (d_V) CHECK_CUDA(cudaFree(d_V));
    if (d_U) CHECK_CUDA(cudaFree(d_U));
    if (d_S) CHECK_CUDA(cudaFree(d_S));
    if (d_A) CHECK_CUDA(cudaFree(d_A));

    CHECK_CUSOLVER(cusolverDnDestroyParams(params));
    CHECK_CUSOLVER(cusolverDnDestroy(handle));
}