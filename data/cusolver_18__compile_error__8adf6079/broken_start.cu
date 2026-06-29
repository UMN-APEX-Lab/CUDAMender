#include "compute_eigenvalues.h"
#include "cusolver_helpers.h"
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <vector>

void compute_eigenvalues_and_vectors(int64_t n, const std::vector<double> &A,
                                     std::vector<cuDoubleComplex> &W,
                                     std::vector<double> &VR) {
    cusolverDnHandle_t handle;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    // Allocate device memory
    double *d_A = nullptr;
    cuDoubleComplex *d_W = nullptr;
    double *d_VR = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, n * sizeof(cuDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_VR, n * n * sizeof(double)));

    // Copy A to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Query workspace size
    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    
    CUSOLVER_CHECK(cusolverDnXgeev_bufferSize(
        handle,
        params,
        /*jobvl*/ 'N',
        /*jobvr*/ 'V',
        n,
        CUDA_R_64F,
        d_A,
        n,
        CUDA_R_64F,
        d_W,
        CUDA_R_64F,
        nullptr, // VL
        n,
        CUDA_R_64F,
        d_VR,
        n,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    // Allocate workspaces
    void *d_work = nullptr;
    void *h_work = nullptr;
    if (workspaceInBytesOnDevice > 0) {
        CUDA_CHECK(cudaMalloc(&d_work, workspaceInBytesOnDevice));
    }
    if (workspaceInBytesOnHost > 0) {
        h_work = malloc(workspaceInBytesOnHost);
    }

    // Device info for error code
    int *d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    // Execute eigenvalue solver
    CUSOLVER_CHECK(cusolverDnXgeev(
        handle,
        params,
        /*jobvl*/ 'N',
        /*jobvr*/ 'V',
        n,
        CUDA_R_64F,
        d_A,
        n,
        CUDA_R_64F,
        d_W,
        CUDA_R_64F,
        nullptr, // VL
        n,
        CUDA_R_64F,
        d_VR,
        n,
        CUDA_R_64F,
        d_work,
        workspaceInBytesOnDevice,
        h_work,
        workspaceInBytesOnHost,
        d_info));

    // Check info
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        // Clean up before throwing
        cudaFree(d_info);
        cudaFree(d_work);
        free(h_work);
        cudaFree(d_VR);
        cudaFree(d_W);
        cudaFree(d_A);
        cusolverDnDestroyParams(params);
        cusolverDnDestroy(handle);
        // Return without modifying outputs on failure
        return;
    }

    // Copy results back to host
    W.resize(n);
    VR.resize(n * n);
    CUDA_CHECK(cudaMemcpy(W.data(), d_W, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(VR.data(), d_VR, n * n * sizeof(double), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_info);
    cudaFree(d_work);
    free(h_work);
    cudaFree(d_VR);
    cudaFree(d_W);
    cudaFree(d_A);
    cusolverDnDestroyParams(params);
    cusolverDnDestroy(handle);
}