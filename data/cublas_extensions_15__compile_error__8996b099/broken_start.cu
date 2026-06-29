#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include "scale_vector.h"

void scale_vector(int n, double alpha,
                  const std::vector<double>& in,
                  std::vector<double>& out) {
    if (n <= 0 || in.size() < static_cast<size_t>(n) || out.size() < static_cast<size_t>(n)) {
        return;
    }

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return;
    }

    // Create CUDA stream
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        cublasDestroy(handle);
        return;
    }

    // Set stream for cuBLAS operations
    cublasSetStream(handle, stream);

    // Allocate device memory
    double *d_in = nullptr;
    double *d_out = nullptr;
    err = cudaMalloc(&d_in, n * sizeof(double));
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        return;
    }
    err = cudaMalloc(&d_out, n * sizeof(double));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        return;
    }

    // Copy input data to device
    err = cudaMemcpyAsync(d_in, in.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        return;
    }

    // Scale vector using cublasScalEx
    status = cublasScalEx(handle,
                          n,
                          &alpha,
                          d_in,
                          1,
                          d_out,
                          1,
                          CUDA_R_64F,
                          CUDA_R_64F,
                          CUDA_R_64F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        return;
    }

    // Copy result back to host
    err = cudaMemcpyAsync(out.data(), d_out, n * sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        return;
    }

    // Synchronize stream to ensure completion
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaStreamDestroy(stream);
        cublasDestroy(handle);
        return;
    }

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    cublasDestroy(handle);
}