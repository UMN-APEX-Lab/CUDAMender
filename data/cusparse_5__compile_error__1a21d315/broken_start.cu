#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(cudaGetErrorString(err)); \
        } \
    } while (0)

#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            throw std::runtime_error("cuSPARSE error"); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error("cuBLAS error"); \
        } \
    } while (0)

void solveGpsvInterleavedBatch(
    int n, int batchSize, const std::vector<float>& h_S,
    const std::vector<float>& h_L, const std::vector<float>& h_M,
    const std::vector<float>& h_U, const std::vector<float>& h_W,
    const std::vector<float>& h_B, std::vector<float>& h_X) {
    
    int total_size = n * batchSize;
    
    if (h_S.size() != static_cast<size_t>(total_size) || 
        h_L.size() != static_cast<size_t>(total_size) ||
        h_M.size() != static_cast<size_t>(total_size) || 
        h_U.size() != static_cast<size_t>(total_size) ||
        h_W.size() != static_cast<size_t>(total_size) || 
        h_B.size() != static_cast<size_t>(total_size)) {
        throw std::invalid_argument("Input vector sizes do not match n * batchSize");
    }

    cusparseHandle_t cusparse_handle;
    cublasHandle_t cublas_handle;
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    float *d_S, *d_L, *d_M, *d_U, *d_W, *d_B;
    float *d_X;
    float *d_S_interleaved, *d_L_interleaved, *d_M_interleaved;
    float *d_U_interleaved, *d_W_interleaved, *d_B_interleaved, *d_X_interleaved;
    
    CUDA_CHECK(cudaMalloc(&d_S, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_L, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_M, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, total_size * sizeof(float)));
    
    CUDA_CHECK(cudaMalloc(&d_S_interleaved, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_L_interleaved, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_M_interleaved, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_U_interleaved, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_interleaved, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_interleaved, total_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X_interleaved, total_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_S, h_S.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_L, h_L.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_U, h_U.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W, h_W.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    CUBLAS_CHECK(cublasSgeam(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, batchSize,
        &alpha, d_S, n,
        &beta, nullptr, n,
        d_S_interleaved, batchSize));
    
    CUBLAS_CHECK(cublasSgeam(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, batchSize,
        &alpha, d_L, n,
        &beta, nullptr, n,
        d_L_interleaved, batchSize));
    
    CUBLAS_CHECK(cublasSgeam(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, batchSize,
        &alpha, d_M, n,
        &beta, nullptr, n,
        d_M_interleaved, batchSize));
    
    CUBLAS_CHECK(cublasSgeam(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, batchSize,
        &alpha, d_U, n,
        &beta, nullptr, n,
        d_U_interleaved, batchSize));
    
    CUBLAS_CHECK(cublasSgeam(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, batchSize,
        &alpha, d_W, n,
        &beta, nullptr, n,
        d_W_interleaved, batchSize));
    
    CUBLAS_CHECK(cublasSgeam(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, batchSize,
        &alpha, d_B, n,
        &beta, nullptr, n,
        d_B_interleaved, batchSize));

    cusparseGpsvInterleavedBatchAlg_t alg = CUSPARSE_GPSV_INTERLEAVED_BATCH_ALG_DEFAULT;
    CUSPARSE_CHECK(cusparseSgpsvInterleavedBatch(
        cusparse_handle,
        n,
        batchSize,
        d_L_interleaved,
        d_S_interleaved,
        d_U_interleaved,
        d_W_interleaved,
        d_M_interleaved,
        d_B_interleaved,
        d_X_interleaved,
        n,
        alg));

    CUBLAS_CHECK(cublasSgeam(cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        batchSize, n,
        &alpha, d_X_interleaved, n,
        &beta, nullptr, n,
        d_X, batchSize));

    h_X.resize(total_size);
    CUDA_CHECK(cudaMemcpy(h_X.data(), d_X, total_size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_S));
    CUDA_CHECK(cudaFree(d_L));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_X));
    
    CUDA_CHECK(cudaFree(d_S_interleaved));
    CUDA_CHECK(cudaFree(d_L_interleaved));
    CUDA_CHECK(cudaFree(d_M_interleaved));
    CUDA_CHECK(cudaFree(d_U_interleaved));
    CUDA_CHECK(cudaFree(d_W_interleaved));
    CUDA_CHECK(cudaFree(d_B_interleaved));
    CUDA_CHECK(cudaFree(d_X_interleaved));

    CUSPARSE_CHECK(cusparseDestroy(cusparse_handle));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}