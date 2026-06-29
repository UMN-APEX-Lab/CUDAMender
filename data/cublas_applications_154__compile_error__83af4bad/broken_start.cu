#include "conjugate_gradient.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-10

extern "C"
void k_conjugateGradientKernel(cublasHandle_t cublasHandle, int matrixSize,
                               double* residual_d, double* matrixA_d,
                               double* searchDir_d, double* vectorX_d,
                               double* productAdir_d)
{
    int n = matrixSize;
    double one = 1.0;
    double zero = 0.0;
    double alpha, beta;
    
    // Allocate temporary device memory for dot product results
    double* d_dot_r = nullptr;
    cudaMalloc(&d_dot_r, sizeof(double));
    double* d_dot_ap = nullptr;
    cudaMalloc(&d_dot_ap, sizeof(double));
    
    // Compute initial residual norm squared (r^T * r)
    cublasDdot(cublasHandle, n, residual_d, 1, residual_d, 1, d_dot_r);
    double r_norm_sq;
    cudaMemcpy(&r_norm_sq, d_dot_r, sizeof(double), cudaMemcpyDeviceToHost);
    
    for (int iter = 0; iter < MAX_ITERATIONS; iter++)
    {
        // Compute A * searchDir_d -> productAdir_d
        // matrixA_d is in column-major format
        // A is n x n, searchDir_d is n x 1
        // productAdir_d = A * searchDir_d
        cublasDgemv(cublasHandle, CUBLAS_OP_N, n, n, &one, matrixA_d, n, searchDir_d, 1, &zero, productAdir_d, 1);
        
        // Compute p^T * (A * p)
        cublasDdot(cublasHandle, n, searchDir_d, 1, productAdir_d, 1, d_dot_ap);
        double pAp;
        cudaMemcpy(&pAp, d_dot_ap, sizeof(double), cudaMemcpyDeviceToHost);
        
        // Compute alpha = (r^T * r) / (p^T * A * p)
        alpha = r_norm_sq / pAp;
        
        // Update x: x = x + alpha * p
        cublasDaxpy(cublasHandle, n, &alpha, searchDir_d, 1, vectorX_d, 1);
        
        // Update r: r = r - alpha * (A * p)
        double neg_alpha = -alpha;
        cublasDaxpy(cublasHandle, n, &neg_alpha, productAdir_d, 1, residual_d, 1);
        
        // Compute new residual norm squared
        cublasDdot(cublasHandle, n, residual_d, 1, residual_d, 1, d_dot_r);
        double r_new_norm_sq;
        cudaMemcpy(&r_new_norm_sq, d_dot_r, sizeof(double), cudaMemcpyDeviceToHost);
        
        // Check convergence AFTER updating the residual
        if (r_new_norm_sq < TOLERANCE * TOLERANCE)
        {
            break;
        }
        
        // Compute beta = (r_new^T * r_new) / (r_old^T * r_old)
        beta = r_new_norm_sq / r_norm_sq;
        
        // Update search direction: p = r_new + beta * p_old
        // First compute beta * p
        cublasDscal(cublasHandle, n, &beta, searchDir_d, 1);
        // Then add r_new to p: p = r_new + p
        cublasDaxpy(cublasHandle, n, &one, residual_d, 1, searchDir_d, 1);
        
        // Update r_norm_sq for next iteration
        r_norm_sq = r_new_norm_sq;
    }
    
    // Free temporary device memory
    cudaFree(d_dot_r);
    cudaFree(d_dot_ap);
}