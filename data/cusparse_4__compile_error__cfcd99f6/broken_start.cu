#include <cuda_runtime.h>
#include <cusparse.h>
#include <vector>
#include "gather.h"

void gather(int size, int nnz, const std::vector<int>& hX_indices,
            const std::vector<float>& hY, std::vector<float>& hX_values) {
    cusparseHandle_t handle;
    cusparseStatus_t status;

    status = cusparseCreate(&handle);

    int *dX_indices;
    float *dY, *dX_values;

    cudaMalloc((void**)&dX_indices, nnz * sizeof(int));
    cudaMalloc((void**)&dY, size * sizeof(float));
    cudaMalloc((void**)&dX_values, nnz * sizeof(float));

    cudaMemcpy(dX_indices, hX_indices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, hY.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;

    status = cusparseGather(
        handle,
        nnz,
        &alpha,
        dY,
        dX_indices,
        dX_values,
        size,
        CUDA_R_32F,
        CUDA_R_32I
    );

    hX_values.resize(nnz);
    cudaMemcpy(hX_values.data(), dX_values, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dX_indices);
    cudaFree(dY);
    cudaFree(dX_values);
    cusparseDestroy(handle);
}