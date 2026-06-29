#include "dense_to_sparse.h"
#include "cusparse_helpers.h"

#include <cusparse.h>
#include <cuda_runtime.h>

#include <vector>

void dense_to_sparse_blocked_ell(int num_rows, int num_cols, int ell_blk_size,
                                 int ell_width,
                                 const std::vector<float> &h_dense,
                                 const std::vector<int> &h_ell_columns,
                                 std::vector<float> &h_ell_values) {
    cusparseHandle_t handle = nullptr;
    cusparseDnMatDescr_t matA = nullptr;
    cusparseSpMatDescr_t matB = nullptr;

    float *d_dense = nullptr;
    int *d_ell_columns = nullptr;
    float *d_ell_values = nullptr;
    void *d_buffer = nullptr;

    const int64_t rows = static_cast<int64_t>(num_rows);
    const int64_t cols = static_cast<int64_t>(num_cols);
    const int64_t block_size = static_cast<int64_t>(ell_blk_size);
    const int64_t ell_cols = static_cast<int64_t>(ell_width);

    const int64_t dense_size = rows * cols;
    const int64_t num_block_rows = rows / block_size;
    const int64_t ell_col_ind_size = num_block_rows * ell_cols;
    const int64_t ell_val_size = num_block_rows * ell_cols * block_size * block_size;

    h_ell_values.resize(static_cast<size_t>(ell_val_size));

    CHECK_CUSPARSE(cusparseCreate(&handle));

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_dense),
                          static_cast<size_t>(dense_size) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_ell_columns),
                          static_cast<size_t>(ell_col_ind_size) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_ell_values),
                          static_cast<size_t>(ell_val_size) * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_dense, h_dense.data(),
                          static_cast<size_t>(dense_size) * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ell_columns, h_ell_columns.data(),
                          static_cast<size_t>(ell_col_ind_size) * sizeof(int),
                          cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseCreateDnMat(&matA,
                                       rows,
                                       cols,
                                       cols,
                                       d_dense,
                                       CUDA_R_32F,
                                       CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSE(cusparseCreateBlockedEll(&matB,
                                            rows,
                                            cols,
                                            block_size,
                                            ell_cols,
                                            d_ell_columns,
                                            d_ell_values,
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO,
                                            CUDA_R_32F));

    size_t buffer_size = 0;
    CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(
        handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &buffer_size));

    CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size));

    CHECK_CUSPARSE(cusparseDenseToSparse_analysis(
        handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));

    CHECK_CUSPARSE(cusparseDenseToSparse_convert(
        handle, matA, matB, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, d_buffer));

    CHECK_CUDA(cudaMemcpy(h_ell_values.data(), d_ell_values,
                          static_cast<size_t>(ell_val_size) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    if (d_buffer) CHECK_CUDA(cudaFree(d_buffer));
    if (matB) CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    if (matA) CHECK_CUSPARSE(cusparseDestroyDnMat(matA));
    if (d_ell_values) CHECK_CUDA(cudaFree(d_ell_values));
    if (d_ell_columns) CHECK_CUDA(cudaFree(d_ell_columns));
    if (d_dense) CHECK_CUDA(cudaFree(d_dense));
    if (handle) CHECK_CUSPARSE(cusparseDestroy(handle));
}