#include "eigen_decomposition.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline void check_cuda(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error at ") + expr + ": " +
                                 cudaGetErrorString(status));
    }
}

inline void check_cusolver(cusolverStatus_t status, const char* expr) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuSOLVER error at ") + expr +
                                 ": status=" + std::to_string(static_cast<int>(status)));
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr)
#define CHECK_CUSOLVER(expr) check_cusolver((expr), #expr)

struct CusolverHandle {
    cusolverDnHandle_t handle{};
    CusolverHandle() { CHECK_CUSOLVER(cusolverDnCreate(&handle)); }
    ~CusolverHandle() { if (handle) cusolverDnDestroy(handle); }
};

struct CusolverParams {
    cusolverDnParams_t params{};
    CusolverParams() { CHECK_CUSOLVER(cusolverDnCreateParams(&params)); }
    ~CusolverParams() { if (params) cusolverDnDestroyParams(params); }
};

template <typename T>
struct DeviceBuffer {
    T* ptr{};
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) {
        if (count > 0) {
            CHECK_CUDA(cudaMalloc(&ptr, count * sizeof(T)));
        }
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct RawDeviceBuffer {
    void* ptr{};
    RawDeviceBuffer() = default;
    explicit RawDeviceBuffer(size_t bytes) {
        if (bytes > 0) {
            CHECK_CUDA(cudaMalloc(&ptr, bytes));
        }
    }
    ~RawDeviceBuffer() { if (ptr) cudaFree(ptr); }
    RawDeviceBuffer(const RawDeviceBuffer&) = delete;
    RawDeviceBuffer& operator=(const RawDeviceBuffer&) = delete;
};

}  // namespace

void compute_eigen_decomposition(int m, const std::vector<double>& A,
                                 std::vector<double>& W, std::vector<double>& V) {
    if (m < 0) {
        throw std::invalid_argument("m must be non-negative");
    }
    const size_t n = static_cast<size_t>(m);
    const size_t matrix_elems = n * n;

    if (A.size() != matrix_elems) {
        throw std::invalid_argument("A size must be m*m");
    }

    W.assign(n, 0.0);
    V.assign(matrix_elems, 0.0);

    if (m == 0) {
        return;
    }

    std::vector<double> hA_col_major(matrix_elems);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            hA_col_major[static_cast<size_t>(j) * n + static_cast<size_t>(i)] =
                A[static_cast<size_t>(i) * n + static_cast<size_t>(j)];
        }
    }

    CusolverHandle solver;
    CusolverParams params;

    DeviceBuffer<double> d_A(matrix_elems);
    DeviceBuffer<double> d_W(n);
    DeviceBuffer<int> d_info(1);

    CHECK_CUDA(cudaMemcpy(d_A.ptr, hA_col_major.data(),
                          matrix_elems * sizeof(double),
                          cudaMemcpyHostToDevice));

    size_t workspace_device_bytes = 0;
    size_t workspace_host_bytes = 0;

    CHECK_CUSOLVER(cusolverDnXsyevd_bufferSize(
        solver.handle,
        params.params,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        static_cast<int64_t>(m),
        CUDA_R_64F,
        d_A.ptr,
        static_cast<int64_t>(m),
        CUDA_R_64F,
        d_W.ptr,
        CUDA_R_64F,
        &workspace_device_bytes,
        &workspace_host_bytes));

    RawDeviceBuffer d_work(workspace_device_bytes);
    std::vector<unsigned char> h_work(workspace_host_bytes);

    CHECK_CUSOLVER(cusolverDnXsyevd(
        solver.handle,
        params.params,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        static_cast<int64_t>(m),
        CUDA_R_64F,
        d_A.ptr,
        static_cast<int64_t>(m),
        CUDA_R_64F,
        d_W.ptr,
        CUDA_R_64F,
        d_work.ptr,
        workspace_device_bytes,
        h_work.empty() ? nullptr : h_work.data(),
        workspace_host_bytes,
        d_info.ptr));

    int info = 0;
    std::vector<double> hV_col_major(matrix_elems);

    CHECK_CUDA(cudaMemcpy(&info, d_info.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(W.data(), d_W.ptr, n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hV_col_major.data(), d_A.ptr,
                          matrix_elems * sizeof(double),
                          cudaMemcpyDeviceToHost));

    if (info < 0) {
        throw std::runtime_error("cusolverDnXsyevd: invalid parameter at position " +
                                 std::to_string(-info));
    }
    if (info > 0) {
        throw std::runtime_error("cusolverDnXsyevd did not converge, info=" +
                                 std::to_string(info));
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            V[static_cast<size_t>(i) * n + static_cast<size_t>(j)] =
                hV_col_major[static_cast<size_t>(j) * n + static_cast<size_t>(i)];
        }
    }
}