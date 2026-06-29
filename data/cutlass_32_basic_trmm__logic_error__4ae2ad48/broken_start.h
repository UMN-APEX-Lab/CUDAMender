#pragma once
#include <cuda_runtime.h>

// Double-precision Triangular Matrix Multiply (TRMM), column-major layout
// C = alpha * A * B
// A is M x M, lower-triangular
// B is M x N
// C is M x N

// Tile dimensions - tuned for double precision
constexpr int TRMM_TILE_M = 64;
constexpr int TRMM_TILE_N = 16;

// Shared memory tile sizes
constexpr int TRMM_SMEM_M = 64;
constexpr int TRMM_SMEM_N = 16;

__global__ void dtrmm_kernel(
    int M, int N,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double *C, int ldc)
{
    // Block indices
    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;
    
    // Thread indices within block
    const int threadRow = threadIdx.x;
    const int threadCol = threadIdx.y;
    
    // Global position in C
    const int row = blockRow * TRMM_TILE_M + threadRow;
    const int col = blockCol * TRMM_TILE_N + threadCol;
    
    // Shared memory for A (lower triangular, so we load by tiles)
    __shared__ double s_A[TRMM_SMEM_M][TRMM_SMEM_M];
    // Shared memory for B
    __shared__ double s_B[TRMM_SMEM_M][TRMM_SMEM_N];
    
    // Accumulator
    double sum = 0.0;
    
    // Number of tiles needed for the inner dimension (M)
    const int numTiles = (M + TRMM_SMEM_M - 1) / TRMM_SMEM_M;
    
    // Each thread computes one element of C
    // We iterate over tiles of A (and corresponding rows of B)
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        // A is lower triangular: A[i,k] is valid only if i >= k
        // We need A[row, t*TRMM_SMEM_M + k] for k in [0, TRMM_SMEM_M)
        
        // Collaborative loading of A tile
        // Each thread loads multiple elements
        #pragma unroll
        for (int i = threadRow; i < TRMM_SMEM_M; i += TRMM_TILE_M) {
            #pragma unroll
            for (int j = threadCol; j < TRMM_SMEM_M; j += TRMM_TILE_N) {
                int globalRow = blockRow * TRMM_TILE_M + i;
                int globalCol = t * TRMM_SMEM_M + j;
                
                double val = 0.0;
                // Only load if within bounds and lower triangular
                if (globalRow < M && globalCol < M && globalRow >= globalCol) {
                    val = A[globalRow + globalCol * lda];
                }
                s_A[i][j] = val;
            }
        }
        
        // Load tile of B into shared memory
        // B[k, col] where k in [t*TRMM_SMEM_M, (t+1)*TRMM_SMEM_M)
        #pragma unroll
        for (int i = threadRow; i < TRMM_SMEM_M; i += TRMM_TILE_M) {
            #pragma unroll
            for (int j = threadCol; j < TRMM_SMEM_N; j += TRMM_TILE_N) {
                int globalRow = t * TRMM_SMEM_M + i;
                int globalCol = blockCol * TRMM_TILE_N + j;
                
                double val = 0.0;
                if (globalRow < M && globalCol < N) {
                    val = B[globalRow + globalCol * ldb];
                }
                s_B[i][j] = val;
            }
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        // C[row, col] += sum over k of A[row, k] * B[k, col]
        // In this tile, k ranges from t*TRMM_SMEM_M to min((t+1)*TRMM_SMEM_M, M)
        
        // The actual row within the tile
        int localRow = threadRow;
        // The actual col within the tile  
        int localCol = threadCol;
        
        // Only compute if this thread has valid output position
        if (row < M && col < N) {
            // Determine valid range of k for this tile
            int kStart = t * TRMM_SMEM_M;
            int kEnd = min(kStart + TRMM_SMEM_M, M);
            
            // For lower triangular A, we need row >= k
            // So k <= row, meaning we only sum k from kStart to min(kEnd, row+1)
            int kMax = min(kEnd, row + 1);
            
            for (int k = kStart; k < kMax; k++) {
                int localK = k - kStart;
                // A[row, k] is at s_A[localRow][localK] (note: row = blockRow*TRMM_TILE_M + localRow)
                // But wait, we loaded A with globalRow, so need to check
                // Actually s_A is indexed by local indices within the tile
                
                // We need: A[row, k] where row is global
                // s_A was loaded with: globalRow = blockRow*TRMM_TILE_M + i, so i = globalRow - blockRow*TRMM_TILE_M
                // For our thread, row = blockRow*TRMM_TILE_M + threadRow, so localRow = threadRow
                
                // However, s_A[i][j] corresponds to A[blockRow*TRMM_TILE_M + i][t*TRMM_SMEM_M + j]
                // So A[row][k] = s_A[threadRow][k - t*TRMM_SMEM_M] = s_A[localRow][localK]
                
                // And B[k][col] = s_B[k - t*TRMM_SMEM_M][threadCol] = s_B[localK][localCol]
                
                sum += s_A[localRow][localK] * s_B[localK][localCol];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row + col * ldc] = alpha * sum;
    }
}

// Alternative kernel with better memory coalescing - using 1D thread blocks
__global__ void dtrmm_kernel_v2(
    int M, int N,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double *C, int ldc)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= M || col >= N) return;
    
    double sum = 0.0;
    
    // For lower triangular A: sum over k from 0 to row (inclusive)
    for (int k = 0; k <= row; k++) {
        sum += A[row + k * lda] * B[k + col * ldb];
    }
    
    C[row + col * ldc] = alpha * sum;
}

// Optimized kernel with shared memory and better tiling
template<int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void dtrmm_kernel_optimized(
    int M, int N,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double *C, int ldc)
{
    __shared__ double s_A[BLOCK_M][BLOCK_K];
    __shared__ double s_B[BLOCK_K][BLOCK_N];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int row = bx * BLOCK_M + tx;
    const int col = by * BLOCK_N + ty;
    
    double sum = 0.0;
    
    // Number of K tiles
    const int numKTiles = (M + BLOCK_K - 1) / BLOCK_K;
    
    for (int kt = 0; kt < numKTiles; kt++) {
        // Load A tile: A[row, kt*BLOCK_K + ty] for ty in [0, BLOCK_K)
        // Each thread loads one element
        // Note: A is lower triangular, so we need row >= k
        
        // Collaborative load of A
        for (int i = tx; i < BLOCK_M; i += blockDim.x) {
            for (int j = ty; j < BLOCK_K; j += blockDim.y) {
                int globalRow = bx * BLOCK_M + i;
                int globalK = kt * BLOCK_K + j;
                
                double val = 0.0;
                if (globalRow < M && globalK < M && globalRow >= globalK) {
                    val = A[globalRow + globalK * lda];
                }
                s_A[i][j] = val;
            }
        }
        
        // Load B tile: B[kt*BLOCK_K + tx, col] 
        // Actually B[k, col] where k in [kt*BLOCK_K, (kt+1)*BLOCK_K)
        for (int i = tx; i < BLOCK_K; i += blockDim.x) {
            for (int j = ty; j < BLOCK_N; j += blockDim.y) {
                int globalK = kt * BLOCK_K + i;
                int globalCol = by * BLOCK_N + j;
                
                double val = 0.0;
                if (globalK < M && globalCol < N) {
                    val = B[globalK + globalCol * ldb];
                }
                s_B[i][j] = val;
            }
        }
        
        __syncthreads();
        
        // Compute on this tile
        if (row < M && col < N) {
            int kStart = kt * BLOCK_K;
            int kEnd = min(kStart + BLOCK_K, M);
            int kMax = min(kEnd, row + 1); // lower triangular constraint
            
            for (int k = kStart; k < kMax; k++) {
                int localK = k - kStart;
                sum += s_A[tx][localK] * s_B[localK][ty];
            }
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row + col * ldc] = alpha * sum;
    }
}

// Simple but correct kernel for small sizes
__global__ void dtrmm_simple(
    int M, int N,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double *C, int ldc)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= M || col >= N) return;
    
    double sum = 0.0;
    // Lower triangular: only sum k where k <= row
    for (int k = 0; k <= row && k < M; k++) {
        sum += A[row + k * lda] * B[k + col * ldb];
    }
    
    C[row + col * ldc] = alpha * sum;
}

// Medium-optimized kernel with register blocking
template<int TM, int TN>
__global__ void dtrmm_kernel_registers(
    int M, int N,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double *C, int ldc)
{
    // Each thread computes TM x TN output elements
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Thread position in terms of output elements
    const int threadRow = (ty * blockDim.x + tx) / TN;
    const int threadCol = (ty * blockDim.x + tx) % TN;
    
    // Actually, let's use a simpler 2D thread layout
    // Re-interpret: blockDim.x threads along M, blockDim.y along N
    // But each thread computes a tile
    
    const int rowStart = bx * blockDim.x * TM + tx * TM;
    const int colStart = by * blockDim.y * TN + ty * TN;
    
    double accum[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            accum[i][j] = 0.0;
        }
    }
    
    // Iterate over K dimension
    for (int k = 0; k < M; k++) {
        // Load A elements for this k
        double aVals[TM];
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            int row = rowStart + i;
            aVals[i] = (row < M && row >= k) ? A[row + k * lda] : 0.0;
        }
        
        // Load B elements for this k
        double bVals[TN];
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int col = colStart + j;
            bVals[j] = (col < N) ? B[k + col * ldb] : 0.0;
        }
        
        // Multiply-accumulate
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                accum[i][j] += aVals[i] * bVals[j];
            }
        }
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int row = rowStart + i;
            int col = colStart + j;
            if (row < M && col < N) {
                C[row + col * ldc] = alpha * accum[i][j];
            }
        }
    }
}

// Main optimized kernel using shared memory with proper coalescing
template<int BM, int BN, int BK>
__global__ void dtrmm_kernel_smem(
    int M, int N,
    double alpha,
    const double *A, int lda,
    const double *B, int ldb,
    double *C, int ldc)
{
    __shared__ double s_A[BM][BK + 1];  // +1 to avoid bank conflicts
    __shared__ double s_B[BK][BN + 1];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Thread computes output at (row, col)
    const int row = bx * BM + tx;
    const int col = by * BN + ty;
    
    double sum = 0.0;
    
    const int numKTiles = (M + BK - 1) / BK;
    
    for (int kt = 0; kt < numKTiles; kt++) {
        // Load A: A[bx*BM + tx, kt*BK + ty] with strided access for coalescing
        // Each thread loads one element, but we need collaborative loading
        
        // Load A tile into shared memory
        // A is M x M, lower triangular
        for (int i = ty; i < BM; i += blockDim.y) {
            for (int j = tx; j < BK; j += blockDim.x) {
                int globalRow = bx * BM + i;
                int globalK = kt * BK + j;
                
                double val = 0.0;
                if (globalRow < M && globalK < M && globalRow >= globalK) {
                    val = A[globalRow + globalK * lda];
                }
                s_A[i][j] = val;
            }
        }
        
        // Load B tile: B[kt*BK + tx, by*BN + ty]
        for (int i = ty; i < BK; i += blockDim.y) {
            for (int j = tx; j < BN; j += blockDim.x) {
                int globalK = kt * BK + i;
                int globalCol = by * BN + j;
                
                double val = 0.0;
                if (globalK < M && globalCol < N) {
                    val = B[globalK + globalCol * ldb];
                }
                s_B[i][j] = val;
            }
        }
        
        __syncthreads();
        
        // Compute on this tile
        if (row < M && col < N) {
            int kStart = kt * BK;
            int kEnd = min(kStart + BK, M);
            // Lower triangular: only k <= row contributes
            int kMax = min(kEnd, row + 1);
            
            // Unroll for performance
            #pragma unroll 4
            for (int k = kStart; k < kMax; k++) {
                int localK = k - kStart;
                sum += s_A[tx][localK] * s_B[localK][ty];
            }
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row + col * ldc] = alpha * sum;
    }
}

cudaError_t Dtrmm(
    int M, int N,
    double alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double *C, int ldc)
{
    cudaError_t err = cudaSuccess;
    
    // Choose kernel based on problem size
    if (M <= 64 || N <= 64) {
        // Simple kernel for small sizes
        const int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((M + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        dtrmm_simple<<<grid, block>>>(M, N, alpha, A, lda, B, ldb, C, ldc);
    } else {
        // Optimized kernel with shared memory
        // Tuned for double precision
        constexpr int BM = 64;
        constexpr int BN = 16;
        constexpr int BK = 16;
        
        dim3 block(16, 16);  // 256 threads per block
        dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
        
        dtrmm_kernel_smem<BM, BN, BK><<<grid, block>>>(
            M, N, alpha, A, lda, B, ldb, C, ldc);
    }
    
    err = cudaGetLastError();
    return err;
}
