#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "lu_factorization.h"

namespace {

inline void cleanup_all(cusolverDnHandle_t handle,
                        double* d_A,
                        int* d_Ipiv,
                        int* d_info) {
    if (d_info) cudaFree(d_info);
    if (d_Ipiv) cudaFree(d_Ipiv);
    if (d_A) cudaFree(d_A);
    if (handle) cusolverDnDestroy(handle);
}

} // namespace

void lu_factorization(int m,
                      const std::vector<double>& A,
                      std::vector<double>& LU,
                      std::vector<int>& Ipiv,
                      int& info,
                      bool pivot_on) {
    info = 0;
    LU.clear();
    Ipiv.clear();

    if (m < 0) {
        info = -1;
        return;
    }

    const size_t n_elem = static_cast<size_t>(m) * static_cast<size_t>(m);
    if (A.size() != n_elem) {
        info = -2;
        return;
    }

    LU = A;
    if (pivot_on) {
        Ipiv.resize(m);
    }

    if (m == 0) {
        return;
    }

    cusolverDnHandle_t handle = nullptr;
    double* d_A = nullptr;
    int* d_Ipiv = nullptr;
    int* d_info = nullptr;
    double* d_work = nullptr;

    if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
        info = -3;
        return;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_A), n_elem * sizeof(double)) != cudaSuccess) {
        info = -4;
        cleanup_all(handle, d_A, d_Ipiv, d_info);
        return;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)) != cudaSuccess) {
        info = -5;
        cleanup_all(handle, d_A, d_Ipiv, d_info);
        return;
    }

    if (pivot_on) {
        if (cudaMalloc(reinterpret_cast<void**>(&d_Ipiv), m * sizeof(int)) != cudaSuccess) {
            info = -6;
            cleanup_all(handle, d_A, d_Ipiv, d_info);
            return;
        }
    }

    if (cudaMemcpy(d_A, A.data(), n_elem * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) {
        info = -7;
        cleanup_all(handle, d_A, d_Ipiv, d_info);
        return;
    }

    int lwork = 0;
    if (cusolverDnDgetrf_bufferSize(handle, m, m, d_A, m, &lwork) != CUSOLVER_STATUS_SUCCESS) {
        info = -8;
        cleanup_all(handle, d_A, d_Ipiv, d_info);
        return;
    }

    if (cudaMalloc(reinterpret_cast<void**>(&d_work), static_cast<size_t>(lwork) * sizeof(double)) != cudaSuccess) {
        info = -9;
        cleanup_all(handle, d_A, d_Ipiv, d_info);
        return;
    }

    cusolverStatus_t status = cusolverDnDgetrf(
        handle,
        m,
        m,
        d_A,
        m,
        d_work,
        pivot_on ? d_Ipiv : nullptr,
        d_info
    );

    cudaFree(d_work);
    d_work = nullptr;

    if (status != CUSOLVER_STATUS_SUCCESS) {
        info = -10;
        cleanup_all(handle, d_A, d_Ipiv, d_info);
        return;
    }

    if (cudaMemcpy(LU.data(), d_A, n_elem * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        info = -11;
        cleanup_all(handle, d_A, d_Ipiv, d_info);
        return;
    }

    if (pivot_on) {
        if (cudaMemcpy(Ipiv.data(), d_Ipiv, m * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            info = -12;
            cleanup_all(handle, d_A, d_Ipiv, d_info);
            return;
        }
    }

    if (cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        info = -13;
        cleanup_all(handle, d_A, d_Ipiv, d_info);
        return;
    }

    cleanup_all(handle, d_A, d_Ipiv, d_info);
}