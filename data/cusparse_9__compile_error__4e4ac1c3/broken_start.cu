#include <vector>
#include <cstddef>

#include "cuda_helpers.h"
#include "sddmm_bsr.h"

#include <cusparse.h>

void sddmm_bsr_example(int A_num_rows, int A_num_cols,
                       int B_num_rows, int B_num_cols,
                       int row_block_dim, int col_block_dim,
                       const std::vector<float>& hA,
                       const std::vector<float>& hB,
                       const std::vector<int>& hC_boffsets,
                       const std::vector<int>& hC_bcolumns,
                       std::vector<float>& hC_values)
{
    // Validate dimensions
    if (A_num_cols != B_num_rows) {
        return;
    }

    // Create cuSPARSE handle
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Device memory allocation
    float* dA = NULL;
    float* dB = NULL;
    int* dC_boffsets = NULL;
    int* dC_bcolumns = NULL;
    float* dC_values = NULL;

    // Calculate sizes
    size_t A_size = (size_t)A_num_rows * A_num_cols * sizeof(float);
    size_t B_size = (size_t)B_num_rows * B_num_cols * sizeof(float);
    int C_num_blocks = hC_boffsets.back();
    size_t C_offsets_size = hC_boffsets.size() * sizeof(int);
    size_t C_columns_size = hC_bcolumns.size() * sizeof(int);
    size_t C_values_size = (size_t)C_num_blocks * row_block_dim * col_block_dim * sizeof(float);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&dA, A_size));
    CHECK_CUDA(cudaMalloc((void**)&dB, B_size));
    CHECK_CUDA(cudaMalloc((void**)&dC_boffsets, C_offsets_size));
    CHECK_CUDA(cudaMalloc((void**)&dC_bcolumns, C_columns_size));
    CHECK_CUDA(cudaMalloc((void**)&dC_values, C_values_size));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), A_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), B_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC_boffsets, hC_boffsets.data(), C_offsets_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC_bcolumns, hC_bcolumns.data(), C_columns_size, cudaMemcpyHostToDevice));

    // Create matrix descriptors
    cusparseDnMatDescr_t matA = NULL;
    cusparseDnMatDescr_t matB = NULL;
    cusparseSpMatDescr_t matC = NULL;

    CHECK_CUSPARSE(cusparseCreateDnMat(&matA, A_num_rows, A_num_cols, A_num_cols, dA,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, B_num_cols, dB,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Create BSR matrix C
    int C_num_rows = (A_num_rows + row_block_dim - 1) / row_block_dim;
    int C_num_cols = (B_num_cols + col_block_dim - 1) / col_block_dim;

    CHECK_CUSPARSE(cusparseCreateBsr(&matC, C_num_rows, C_num_cols, C_num_blocks,
                                     row_block_dim, col_block_dim,
                                     dC_boffsets, dC_bcolumns, dC_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                     CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Allocate workspace for SDDMM
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSDDMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT,
        &bufferSize));

    void* dBuffer = NULL;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // Set alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform SDDMM
    CHECK_CUSPARSE(cusparseSDDMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SDDMM_ALG_DEFAULT,
        dBuffer));

    // Synchronize
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    hC_values.resize(C_num_blocks * row_block_dim * col_block_dim);
    CHECK_CUDA(cudaMemcpy(hC_values.data(), dC_values, C_values_size, cudaMemcpyDeviceToHost));

    // Cleanup
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matA));
    CHECK_CUDA(cudaFree(dC_values));
    CHECK_CUDA(cudaFree(dC_bcolumns));
    CHECK_CUDA(cudaFree(dC_boffsets));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}