#ifndef CONJUGATE_GRADIENT_H
#define CONJUGATE_GRADIENT_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#define CUBLAS_CHECK(call) \
    { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS Error: %d, line %d\n", status, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    }

#define TOLERANCE 1e-3      // Tolerance for floating-point comparisons
#define MAX_ITERATIONS 1000 // Maximum iterations for Conjugate Gradient

void k_conjugateGradientKernel(cublasHandle_t cublasHandle, int matrixSize, double* residual_d, double* matrixA_d, double* searchDir_d, double* vectorX_d, double* productAdir_d);

#endif // CONJUGATE_GRADIENT_H