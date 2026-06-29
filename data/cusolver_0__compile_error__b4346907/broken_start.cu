#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

void solve_matrix(int N, int nrhs, const std::vector<double> &hA,
                  const std::vector<double> &hB, std::vector<double> &hX) {
    cusolverDnHandle_t handle = nullptr;
    cusolverDnIRSParams_t params = nullptr;
    cusolverDnIRSInfos_t infos = nullptr;
    
    double *dA = nullptr;
    double *dB = nullptr;
    double *dX = nullptr;
    double *dWorkspace = nullptr;
    int *dInfo = nullptr;
    
    size_t dWorkspaceSize = 0;
    
    try {
        // Initialize cuSOLVER handle
        cusolverStatus_t status = cusolverDnCreate(&handle);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cusolverDnCreate failed");
        }
        
        // Create IRS parameters
        status = cusolverDnIRSParamsCreate(&params);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cusolverDnIRSParamsCreate failed");
        }
        
        // Configure IRS parameters for double precision
        status = cusolverDnIRSParamsSetSolverPrecisions(params, 
                                                        CUSOLVER_R_FP64,  // input/output precision
                                                        CUSOLVER_R_FP64); // factorization precision
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cusolverDnIRSParamsSetSolverPrecisions failed");
        }
        
        // Set refinement solver to use standard iterative refinement
        status = cusolverDnIRSParamsSetRefinementSolver(params, 
                                                        CUSOLVER_IRS_REFINE_CLASSICAL);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cusolverDnIRSParamsSetRefinementSolver failed");
        }
        
        // Create IRS info object
        status = cusolverDnIRSInfosCreate(&infos);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cusolverDnIRSInfosCreate failed");
        }
        
        // Allocate device memory
        cudaError_t cudaStatus = cudaMalloc(&dA, sizeof(double) * N * N);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc dA failed");
        }
        
        cudaStatus = cudaMalloc(&dB, sizeof(double) * N * nrhs);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc dB failed");
        }
        
        cudaStatus = cudaMalloc(&dX, sizeof(double) * N * nrhs);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc dX failed");
        }
        
        cudaStatus = cudaMalloc(&dInfo, sizeof(int));
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc dInfo failed");
        }
        
        // Copy host data to device
        cudaStatus = cudaMemcpy(dA, hA.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy hA to dA failed");
        }
        
        cudaStatus = cudaMemcpy(dB, hB.data(), sizeof(double) * N * nrhs, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy hB to dB failed");
        }
        
        // Query workspace size
        status = cusolverDnIRSXgesv_bufferSize(handle, params, N, nrhs, 
                                               &dWorkspaceSize);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cusolverDnIRSXgesv_bufferSize failed");
        }
        
        // Allocate workspace
        cudaStatus = cudaMalloc(&dWorkspace, dWorkspaceSize);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMalloc dWorkspace failed");
        }
        
        // Solve the system using IRS
        int iterCount = 0;
        status = cusolverDnIRSXgesv(handle, params, infos, N, nrhs,
                                    dA, N,
                                    dB, N,
                                    dX, N,
                                    dWorkspace, dWorkspaceSize,
                                    &iterCount, dInfo);
        if (status != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("cusolverDnIRSXgesv failed");
        }
        
        // Check info
        int hInfo = 0;
        cudaStatus = cudaMemcpy(&hInfo, dInfo, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy dInfo to hInfo failed");
        }
        
        if (hInfo != 0) {
            throw std::runtime_error("cusolverDnIRSXgesv returned non-zero info");
        }
        
        // Copy solution back to host
        hX.resize(N * nrhs);
        cudaStatus = cudaMemcpy(hX.data(), dX, sizeof(double) * N * nrhs, cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy dX to hX failed");
        }
        
    } catch (...) {
        // Cleanup on exception
        if (dWorkspace) cudaFree(dWorkspace);
        if (dInfo) cudaFree(dInfo);
        if (dX) cudaFree(dX);
        if (dB) cudaFree(dB);
        if (dA) cudaFree(dA);
        if (infos) cusolverDnIRSInfosDestroy(infos);
        if (params) cusolverDnIRSParamsDestroy(params);
        if (handle) cusolverDnDestroy(handle);
        throw;
    }
    
    // Cleanup
    if (dWorkspace) cudaFree(dWorkspace);
    if (dInfo) cudaFree(dInfo);
    if (dX) cudaFree(dX);
    if (dB) cudaFree(dB);
    if (dA) cudaFree(dA);
    if (infos) cusolverDnIRSInfosDestroy(infos);
    if (params) cusolverDnIRSParamsDestroy(params);
    if (handle) cusolverDnDestroy(handle);
}