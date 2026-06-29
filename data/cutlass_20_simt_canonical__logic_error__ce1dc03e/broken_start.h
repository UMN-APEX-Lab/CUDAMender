#pragma once
#include <cuda_runtime.h>

// SGEMM with mixed layouts:
// A: row-major (M x K)
// B: column-major (K x N) 
// C: row-major (M x N)
// D: row-major (M x N)
// D = alpha * A * B + beta * C

// Use a tile-based approach with shared memory for efficient access patterns

// Tile dimensions - tuned for modern GPUs
// Each thread block computes a TILE_M x TILE_N tile of D
// Each thread computes TM x TN elements
#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

// Number of threads per block
#define BLOCK_DIM_X 16  // TILE_N / TN = 64 / 4 = 16
#define BLOCK_DIM_Y 16  // TILE_M / TM = 64 / 4 = 16

// Elements computed per thread
#define TM 4
#define TN 4

__global__ void simtCanonicalGemmKernel(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float const *C, int ldc,
    float *D, int ldd)
{
    // Block position in grid
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread position within block
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    // Global position of this thread's output tile
    int rowStart = blockRow * TILE_M + threadRow * TM;
    int colStart = blockCol * TILE_N + threadCol * TN;
    
    // Registers for accumulating TM x TN results
    float accum[TM][TN];
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            accum[m][n] = 0.0f;
        }
    }
    
    // Shared memory tiles
    // A tile: TILE_M x TILE_K (row-major in smem)
    // B tile: TILE_K x TILE_N (column-major in smem, but we access it transposed)
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];
    
    // Loop over K in tiles of TILE_K
    for (int kTile = 0; kTile < K; kTile += TILE_K) {
        // Load A tile (row-major) into shared memory
        // Each thread loads multiple elements
        // A is row-major: A[i * lda + k]
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            #pragma unroll
            for (int k = 0; k < TILE_K; k += BLOCK_DIM_X) {
                int loadRow = threadRow * TM + m;
                int loadK = k + threadCol;
                
                float val = 0.0f;
                int globalRow = blockRow * TILE_M + loadRow;
                int globalK = kTile + loadK;
                
                if (globalRow < M && globalK < K) {
                    val = A[globalRow * lda + globalK];
                }
                
                As[loadRow][loadK] = val;
            }
        }
        
        // Load B tile (column-major) into shared memory
        // B is column-major: B[k + j * ldb]
        // We need Bs to be accessible as Bs[k][n] where k is the k-index and n is the n-index
        // For column-major B[k + j * ldb], element at (k, j) is at B[k + j * ldb]
        #pragma unroll
        for (int k = 0; k < TILE_K; k += BLOCK_DIM_Y) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int loadK = k + threadRow;
                int loadCol = threadCol * TN + n;
                
                float val = 0.0f;
                int globalK = kTile + loadK;
                int globalCol = blockCol * TILE_N + loadCol;
                
                if (globalK < K && globalCol < N) {
                    val = B[globalK + globalCol * ldb];
                }
                
                // Store in shared memory: Bs[loadK][loadCol]
                // Need to handle bank conflicts carefully
                Bs[loadK][loadCol] = val;
            }
        }
        
        __syncthreads();
        
        // Compute partial dot products for this tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // Load A values for this thread's rows
            float aVals[TM];
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                aVals[m] = As[threadRow * TM + m][k];
            }
            
            // Load B values for this thread's columns
            float bVals[TN];
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                bVals[n] = Bs[k][threadCol * TN + n];
            }
            
            // Accumulate
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += aVals[m] * bVals[n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to D with alpha and beta scaling
    // D = alpha * (A * B) + beta * C
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int globalRow = rowStart + m;
        if (globalRow >= M) break;
        
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int globalCol = colStart + n;
            if (globalCol >= N) break;
            
            float cVal = 0.0f;
            if (beta != 0.0f) {
                cVal = C[globalRow * ldc + globalCol];
            }
            
            float result = beta * accum[m][n] + alpha * cVal;
            D[globalRow * ldd + globalCol] = result;
        }
    }
}

// Simpler fallback kernel for small sizes
__global__ void simtCanonicalGemmSimpleKernel(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float const *C, int ldc,
    float *D, int ldd)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A is row-major: A[row * lda + k]
            // B is column-major: B[k + col * ldb]
            sum += A[row * lda + k] * B[k + col * ldb];
        }
        
        float cVal = (beta != 0.0f) ? C[row * ldc + col] : 0.0f;
        D[row * ldd + col] = beta * sum + alpha * cVal;
    }
}

cudaError_t SimtCanonicalGemm(
    int M, int N, int K, float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float const *C, int ldc,
    float *D, int ldd)
{
    cudaError_t err = cudaSuccess;
    
    // Choose kernel based on problem size
    // For very small problems, use simple kernel
    if (M <= 64 || N <= 64 || K <= 16) {
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);
        
        simtCanonicalGemmSimpleKernel<<<gridDim, blockDim>>>(
            M, N, K, alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc,
            D, ldd);
    } else {
        // Use tiled kernel for larger problems
        dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
        dim3 gridDim((N + TILE_N - 1) / TILE_N,
                     (M + TILE_M - 1) / TILE_M);
        
        simtCanonicalGemmKernel<<<gridDim, blockDim>>>(
            M, N, K, alpha,
            A, lda,
            B, ldb,
            beta,
            C, ldc,
            D, ldd);
    }
    
    err = cudaGetLastError();
    return err;
}
