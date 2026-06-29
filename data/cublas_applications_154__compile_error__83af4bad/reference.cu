#include "conjugate_gradient.h"
#include <cmath>

void k_conjugateGradientKernel(cublasHandle_t cublasHandle, int matrixSize, double* residual_d, double* matrixA_d, double* searchDir_d, double* vectorX_d, double* productAdir_d) {
    int iterator = 0;
    double oldResidual, newResidual, alpha, beta;

    // Compute oldResidual = residualVector^T * residualVector
    CUBLAS_CHECK(cublasDdot(cublasHandle, matrixSize, residual_d, 1, residual_d, 1, &oldResidual));

    while (iterator < MAX_ITERATIONS) {
        // productAdir = matrixA * searchDirection
        double alphaCublas = 1.0;
        double betaCublas = 0.0;
        CUBLAS_CHECK(cublasDgemv(cublasHandle, CUBLAS_OP_N, matrixSize, matrixSize,
                    &alphaCublas, matrixA_d, matrixSize,
                    searchDir_d, 1,
                    &betaCublas, productAdir_d, 1));

        // Compute alpha = oldResidual / (searchDirection^T * productAdir)
        double pAp;
        CUBLAS_CHECK(cublasDdot(cublasHandle, matrixSize, searchDir_d, 1, productAdir_d, 1, &pAp));
        if (pAp == 0.0)
        {
            break; // Encountered zero denominator in alpha computation.
        }
        alpha = oldResidual / pAp;

        // Update vectorX = vectorX + alpha * searchDirection
        CUBLAS_CHECK(cublasDaxpy(cublasHandle, matrixSize, &alpha, searchDir_d, 1, vectorX_d, 1));

        // Update residualVector = residualVector - alpha * productAdir
        double negAlpha = -1.0 * alpha;
        CUBLAS_CHECK(cublasDaxpy(cublasHandle, matrixSize, &negAlpha, productAdir_d, 1, residual_d, 1));

        // Compute newResidual = residualVector^T * residualVector
        CUBLAS_CHECK(cublasDdot(cublasHandle, matrixSize, residual_d, 1, residual_d, 1, &newResidual));

        // Check for convergence
        if (std::sqrt(newResidual) < TOLERANCE) {
            break;
        }

        // Compute beta = newResidual / oldResidual
        beta = newResidual / oldResidual;

        // Update searchDirection = residualVector + beta * searchDirection
        CUBLAS_CHECK(cublasDscal(cublasHandle, matrixSize, &beta, searchDir_d, 1));                       // searchDirection = beta * searchDirection
        CUBLAS_CHECK(cublasDaxpy(cublasHandle, matrixSize, &alphaCublas, residual_d, 1, searchDir_d, 1)); // searchDirection = searchDirection + residualVector

        // Update oldResidual for next iteration
        oldResidual = newResidual;
        iterator++;
    }
}