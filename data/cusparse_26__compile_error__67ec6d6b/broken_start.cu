#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "include/spsv_csr.h"
#include "include/cusparse_utils.h"

#ifndef CHECK_CUDA
#define CHECK_CUDA(func)                                                                 \
    do {                                                                                 \
        cudaError_t status_ = (func);                                                    \
        if (status_ != cudaSuccess) {                                                    \
            throw std::runtime_error(cudaGetErrorString(status_));                       \
        }                                                                                \
    } while (0)
#endif

#ifndef CHECK_CUSPARSE
#define CHECK_CUSPARSE(func)                                                             \
    do {                                                                                 \
        cusparseStatus_t status_ = (func);                                               \
        if (status_ != CUSPARSE_STATUS_SUCCESS) {                                        \
            throw std::runtime_error("cuSPARSE error");                                  \
        }                                                                                \
    } while (0)
#endif

void spsv_csr_example(int A_num_rows, int A_num_cols, int A_nnz,
                      const std::vector<int>& hA_csrOffsets,
                      const std::vector<int>& hA_columns,
                      const std::vector<float>& hA_values,
                      const std::vector<float>& hX, std::vector<float>& hY) {
    if (A_num_rows < 0 || A_num_cols < 0 || A_nnz < 0) {
        throw std::invalid_argument("Invalid matrix sizes");
    }
    if (static_cast<int>(hA_csrOffsets.size()) != A_num_rows + 1) {
        throw std::invalid_argument("hA_csrOffsets size mismatch");
    }
    if (static_cast<int>(hA_columns.size()) != A_nnz ||
        static_cast<int>(hA_values.size()) != A_nnz) {
        throw std::invalid_argument("CSR column/value size mismatch");
    }
    if (static_cast<int>(hX.size()) != A_num_cols) {
        throw std::invalid_argument("hX size mismatch");
    }
    if (A_num_rows != A_num_cols) {
        throw std::invalid_argument("SpSV requires a square triangular matrix");
    }

    hY.resize(A_num_rows, 0.0f);

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseDnVecDescr_t vecY = nullptr;
    cusparseSpSVDescr_t spsvDescr = nullptr;

    int* dA_csrOffsets = nullptr;
    int* dA_columns = nullptr;
    float* dA_values = nullptr;
    float* dX = nullptr;
    float* dY = nullptr;
    void* dBuffer = nullptr;

    try {
        CHECK_CUSPARSE(cusparseCreate(&handle));

        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dA_csrOffsets),
                              sizeof(int) * hA_csrOffsets.size()));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dA_columns),
                              sizeof(int) * hA_columns.size()));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dA_values),
                              sizeof(float) * hA_values.size()));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dX),
                              sizeof(float) * hX.size()));
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dY),
                              sizeof(float) * hY.size()));

        CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets.data(),
                              sizeof(int) * hA_csrOffsets.size(),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns.data(),
                              sizeof(int) * hA_columns.size(),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dA_values, hA_values.data(),
                              sizeof(float) * hA_values.size(),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dX, hX.data(),
                              sizeof(float) * hX.size(),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(dY, 0, sizeof(float) * hY.size()));

        CHECK_CUSPARSE(cusparseCreateCsr(
            &matA,
            A_num_rows,
            A_num_cols,
            A_nnz,
            dA_csrOffsets,
            dA_columns,
            dA_values,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F));

        CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, CUDA_R_32F));
        CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescr));

        CHECK_CUSPARSE(
            cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE,
                                      &(cusparseFillMode_t){CUSPARSE_FILL_MODE_LOWER},
                                      sizeof(cusparseFillMode_t)));

        CHECK_CUSPARSE(
            cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                      &(cusparseDiagType_t){CUSPARSE_DIAG_TYPE_NON_UNIT},
                                      sizeof(cusparseDiagType_t)));

        float alpha = 1.0f;
        size_t bufferSize = 0;

        CHECK_CUSPARSE(cusparseSpSV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            vecY,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spsvDescr,
            &bufferSize));

        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

        CHECK_CUSPARSE(cusparseSpSV_analysis(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            vecY,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spsvDescr,
            dBuffer));

        CHECK_CUSPARSE(cusparseSpSV_solve(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            vecY,
            CUDA_R_32F,
            CUSPARSE_SPSV_ALG_DEFAULT,
            spsvDescr));

        CHECK_CUDA(cudaMemcpy(hY.data(), dY, sizeof(float) * hY.size(),
                              cudaMemcpyDeviceToHost));

        if (dBuffer) cudaFree(dBuffer);
        if (spsvDescr) cusparseSpSV_destroyDescr(spsvDescr);
        if (vecY) cusparseDestroyDnVec(vecY);
        if (vecX) cusparseDestroyDnVec(vecX);
        if (matA) cusparseDestroySpMat(matA);
        if (dY) cudaFree(dY);
        if (dX) cudaFree(dX);
        if (dA_values) cudaFree(dA_values);
        if (dA_columns) cudaFree(dA_columns);
        if (dA_csrOffsets) cudaFree(dA_csrOffsets);
        if (handle) cusparseDestroy(handle);
    } catch (...) {
        if (dBuffer) cudaFree(dBuffer);
        if (spsvDescr) cusparseSpSV_destroyDescr(spsvDescr);
        if (vecY) cusparseDestroyDnVec(vecY);
        if (vecX) cusparseDestroyDnVec(vecX);
        if (matA) cusparseDestroySpMat(matA);
        if (dY) cudaFree(dY);
        if (dX) cudaFree(dX);
        if (dA_values) cudaFree(dA_values);
        if (dA_columns) cudaFree(dA_columns);
        if (dA_csrOffsets) cudaFree(dA_csrOffsets);
        if (handle) cusparseDestroy(handle);
        throw;
    }
}