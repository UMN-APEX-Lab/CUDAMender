#include "compute_svd.h"
#include "cuda_utils.h"
#include <cusolverDn.h>

void compute_svd(int64_t m, int64_t n, int64_t rank, int64_t p, int64_t iters,
                 std::vector<double>& A, std::vector<double>& S,
                 std::vector<double>& U, std::vector<double>& V) {
    // Create cuSOLVER handle
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    // Create and configure cuSOLVER parameters
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));

    // Set data types for double precision
    cudaDataType dataType = CUDA_R_64F;
    cudaDataType computeType = CUDA_R_64F;

    // Calculate dimensions
    int64_t k = rank;
    int64_t l = rank + p;  // Oversampled rank

    // Leading dimensions (column-major)
    int64_t lda = m;
    int64_t ldu = m;
    int64_t ldv = n;

    // Resize output vectors
    S.resize(k);
    U.resize(m * k);
    V.resize(n * k);

    // Allocate device memory
    double* d_A = nullptr;
    double* d_S = nullptr;
    double* d_U = nullptr;
    double* d_V = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * m * n));
    CUDA_CHECK(cudaMalloc(&d_S, sizeof(double) * k));
    CUDA_CHECK(cudaMalloc(&d_U, sizeof(double) * m * k));
    CUDA_CHECK(cudaMalloc(&d_V, sizeof(double) * n * k));

    // Copy input matrix to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(double) * m * n, cudaMemcpyHostToDevice));

    // Query workspace sizes
    size_t d_workSize = 0;
    size_t h_workSize = 0;

    CUSOLVER_CHECK(cusolverDnXgesvdr_bufferSize(
        handle,
        params,
        'S',  // jobu: economy mode for left singular vectors
        'S',  // jobv: economy mode for right singular vectors
        m,
        n,
        k,
        l,
        iters,
        dataType,
        d_A,
        lda,
        dataType,
        d_S,
        dataType,
        d_U,
        ldu,
        dataType,
        d_V,
        ldv,
        computeType,
        &d_workSize,
        &h_workSize
    ));

    // Allocate workspace
    void* d_work = nullptr;
    void* h_work = nullptr;

    CUDA_CHECK(cudaMalloc(&d_work, d_workSize));
    if (h_workSize > 0) {
        h_work = malloc(h_workSize);
    }

    // Device info for error checking
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Execute randomized SVD
    CUSOLVER_CHECK(cusolverDnXgesvdr(
        handle,
        params,
        'S',  // jobu: economy mode for left singular vectors
        'S',  // jobv: economy mode for right singular vectors
        m,
        n,
        k,
        l,
        iters,
        dataType,
        d_A,
        lda,
        dataType,
        d_S,
        dataType,
        d_U,
        ldu,
        dataType,
        d_V,
        ldv,
        computeType,
        d_work,
        d_workSize,
        h_work,
        h_workSize,
        d_info
    ));

    // Check for errors
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(S.data(), d_S, sizeof(double) * k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(U.data(), d_U, sizeof(double) * m * k, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(V.data(), d_V, sizeof(double) * n * k, cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(d_info));

    if (h_work != nullptr) {
        free(h_work);
    }

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));
}