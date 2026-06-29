#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <vector>
#include <stdexcept>

void compute_svd_batched(int batchSize, int m, int n, 
                         const std::vector<float>& A, 
                         std::vector<float>& S, 
                         std::vector<float>& U, 
                         std::vector<float>& V) {
    
    // Create cuSOLVER handle
    cusolverDnHandle_t handle;
    cusolverStatus_t status = cusolverDnCreate(&handle);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuSOLVER handle");
    }
    
    // Determine singular value count
    int min_dim = std::min(m, n);
    
    // Calculate sizes for batched matrices
    size_t size_A = static_cast<size_t>(batchSize) * m * n;
    size_t size_S = static_cast<size_t>(batchSize) * min_dim;
    size_t size_U = static_cast<size_t>(batchSize) * m * m;
    size_t size_V = static_cast<size_t>(batchSize) * n * n;
    
    // Allocate device memory
    float *d_A = nullptr, *d_S = nullptr, *d_U = nullptr, *d_V = nullptr;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_S, size_S * sizeof(float));
    cudaMalloc(&d_U, size_U * sizeof(float));
    cudaMalloc(&d_V, size_V * sizeof(float));
    
    // Allocate device memory for workspace and info
    float *d_work = nullptr;
    int *d_devInfo = nullptr;
    cudaMalloc(&d_work, 1 * sizeof(float));
    cudaMalloc(&d_devInfo, batchSize * sizeof(int));
    
    // Transfer input data from host to device
    cudaMemcpy(d_A, A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    
    // Compute workspace size for cuSOLVER
    int lwork = 0;
    status = cusolverDnSgesvdaStridedBatched(handle,
                                             CUSOLVER_EIG_MODE_VECTOR,
                                             m, n,
                                             d_A, m,
                                             d_S, min_dim,
                                             d_U, m,
                                             d_V, n,
                                             d_work, lwork,
                                             batchSize,
                                             static_cast<long long int>(m) * n,
                                             static_cast<long long int>(min_dim),
                                             static_cast<long long int>(m) * m,
                                             static_cast<long long int>(n) * n,
                                             d_devInfo);
    
    // Get required workspace size
    if (status == CUSOLVER_STATUS_NOT_SUPPORTED) {
        // Query workspace size
        float* d_temp = nullptr;
        int lwork_needed = 0;
        status = cusolverDnSgesvdaStridedBatched(handle,
                                                 CUSOLVER_EIG_MODE_VECTOR,
                                                 m, n,
                                                 d_A, m,
                                                 d_S, min_dim,
                                                 d_U, m,
                                                 d_V, n,
                                                 d_temp, lwork_needed,
                                                 batchSize,
                                                 static_cast<long long int>(m) * n,
                                                 static_cast<long long int>(min_dim),
                                                 static_cast<long long int>(m) * m,
                                                 static_cast<long long int>(n) * n,
                                                 d_devInfo);
        cudaMalloc(&d_work, (lwork_needed + 1) * sizeof(float));
    } else {
        cudaFree(d_work);
        cudaMalloc(&d_work, 1 * sizeof(float));
    }
    
    // Allocate workspace with proper size
    int lwork_size = 0;
    status = cusolverDnSgesvdaStridedBatched(handle,
                                             CUSOLVER_EIG_MODE_VECTOR,
                                             m, n,
                                             d_A, m,
                                             d_S, min_dim,
                                             d_U, m,
                                             d_V, n,
                                             d_work, lwork_size,
                                             batchSize,
                                             static_cast<long long int>(m) * n,
                                             static_cast<long long int>(min_dim),
                                             static_cast<long long int>(m) * m,
                                             static_cast<long long int>(n) * n,
                                             d_devInfo);
    
    // Actually get workspace size properly
    cudaFree(d_work);
    status = cusolverDnSgesvdaStridedBatched(handle,
                                             CUSOLVER_EIG_MODE_VECTOR,
                                             m, n,
                                             d_A, m,
                                             d_S, min_dim,
                                             d_U, m,
                                             d_V, n,
                                             nullptr, 0,
                                             batchSize,
                                             static_cast<long long int>(m) * n,
                                             static_cast<long long int>(min_dim),
                                             static_cast<long long int>(m) * m,
                                             static_cast<long long int>(n) * n,
                                             d_devInfo);
    cudaMalloc(&d_work, 1 * sizeof(float));
    
    // Compute SVD
    status = cusolverDnSgesvdaStridedBatched(handle,
                                             CUSOLVER_EIG_MODE_VECTOR,
                                             m, n,
                                             d_A, m,
                                             d_S, min_dim,
                                             d_U, m,
                                             d_V, n,
                                             d_work, 1,
                                             batchSize,
                                             static_cast<long long int>(m) * n,
                                             static_cast<long long int>(min_dim),
                                             static_cast<long long int>(m) * m,
                                             static_cast<long long int>(n) * n,
                                             d_devInfo);
    
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cuSOLVER SVD computation failed");
    }
    
    // Transfer results back to host
    cudaMemcpy(S.data(), d_S, size_S * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(U.data(), d_U, size_U * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(V.data(), d_V, size_V * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup device memory
    cudaFree(d_A);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_work);
    cudaFree(d_devInfo);
    
    // Destroy cuSOLVER handle
    cusolverDnDestroy(handle);
}