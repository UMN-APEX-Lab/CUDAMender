#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Half-precision GEMM with row-major layout
// D = alpha * A * B + beta * D
// Accumulation in FP32 for numerical stability

// Tile dimensions - tuned for modern GPUs
constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 32;

// Thread block dimensions
constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;
constexpr int BLOCK_K = 8;

// Number of threads per block
constexpr int THREADS_M = TILE_M / BLOCK_M;  // 8
constexpr int THREADS_N = TILE_N / BLOCK_N;  // 8
constexpr int THREADS_PER_BLOCK = THREADS_M * THREADS_N;  // 64

// Shared memory tile with padding to avoid bank conflicts
constexpr int SMEM_PAD_A = 8;
constexpr int SMEM_PAD_B = 8;

__global__ void gemm_permute_kernel(
    int M, int N, int K, float alpha,
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    float beta,
    __half* __restrict__ D, int ldd)
{
    // Block indices
    const int block_m = blockIdx.x;
    const int block_n = blockIdx.y;
    
    // Thread indices within block
    const int thread_m = threadIdx.x % THREADS_M;
    const int thread_n = threadIdx.x / THREADS_M;
    
    // Starting positions for this block
    const int m_start = block_m * TILE_M;
    const int n_start = block_n * TILE_N;
    
    // Global thread positions
    const int global_m = m_start + thread_m * BLOCK_M;
    const int global_n = n_start + thread_n * BLOCK_N;
    
    // Shared memory tiles
    __shared__ __half smem_A[TILE_K][TILE_M + SMEM_PAD_A];
    __shared__ __half smem_B[TILE_N + SMEM_PAD_B][TILE_K];
    
    // Accumulators in FP32
    float accum[BLOCK_M][BLOCK_N];
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) {
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j++) {
            accum[i][j] = 0.0f;
        }
    }
    
    // Loop over K dimension in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A tile: TILE_K x TILE_M from A (transposed access for coalescing)
        // Each thread loads a portion of the tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k += 2) {
            #pragma unroll
            for (int m = 0; m < BLOCK_M; m += 2) {
                int local_k = k + (threadIdx.x % (TILE_K / 2));
                int local_m = (thread_m * BLOCK_M + m) + (threadIdx.x / (TILE_K / 2)) * 2;
                
                // Ensure we stay within bounds
                if (local_m < TILE_M && local_k < TILE_K) {
                    int global_k = k_tile + local_k;
                    int global_m_load = m_start + local_m;
                    
                    if (global_m_load < M && global_k < K) {
                        __half2 val = *reinterpret_cast<const __half2*>(
                            &A[global_m_load * lda + global_k]);
                        smem_A[local_k][local_m] = val.x;
                        if (local_m + 1 < TILE_M) {
                            smem_A[local_k][local_m + 1] = val.y;
                        }
                    } else {
                        smem_A[local_k][local_m] = __half(0);
                        if (local_m + 1 < TILE_M) {
                            smem_A[local_k][local_m + 1] = __half(0);
                        }
                    }
                }
            }
        }
        
        // Load B tile: TILE_N x TILE_K from B
        #pragma unroll
        for (int n = 0; n < BLOCK_N; n += 2) {
            #pragma unroll
            for (int k = 0; k < TILE_K; k += 2) {
                int local_n = (thread_n * BLOCK_N + n) + ((threadIdx.x % 4) * 2);
                int local_k = (threadIdx.x / 4);
                
                if (local_n < TILE_N && local_k < TILE_K) {
                    int global_k = k_tile + local_k;
                    int global_n_load = n_start + local_n;
                    
                    if (global_n_load < N && global_k < K) {
                        __half2 val = *reinterpret_cast<const __half2*>(
                            &B[global_k * ldb + global_n_load]);
                        smem_B[local_n][local_k] = val.x;
                        if (local_n + 1 < TILE_N) {
                            smem_B[local_n + 1][local_k] = val.y;
                        }
                    } else {
                        smem_B[local_n][local_k] = __half(0);
                        if (local_n + 1 < TILE_N) {
                            smem_B[local_n + 1][local_k] = __half(0);
                        }
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute matrix multiplication on the tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // Load A values for this thread's rows
            __half a_vals[BLOCK_M];
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i++) {
                int a_row = thread_m * BLOCK_M + i;
                if (a_row < TILE_M) {
                    a_vals[i] = smem_A[k][a_row];
                }
            }
            
            // Load B values for this thread's columns
            __half b_vals[BLOCK_N];
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j++) {
                int b_col = thread_n * BLOCK_N + j;
                if (b_col < TILE_N) {
                    b_vals[j] = smem_B[b_col][k];
                }
            }
            
            // Multiply and accumulate
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i++) {
                #pragma unroll
                for (int j = 0; j < BLOCK_N; j++) {
                    accum[i][j] += __half2float(a_vals[i]) * __half2float(b_vals[j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results with alpha and beta scaling
    #pragma unroll
    for (int i = 0; i < BLOCK_M; i++) {
        int out_m = global_m + i;
        if (out_m >= M) break;
        
        #pragma unroll
        for (int j = 0; j < BLOCK_N; j += 2) {
            int out_n = global_n + j;
            if (out_n >= N) break;
            
            float val0 = accum[i][j] * alpha;
            float val1 = (j + 1 < BLOCK_N) ? accum[i][j + 1] * alpha : 0.0f;
            
            if (beta != 0.0f) {
                if (out_n < N) {
                    val0 += beta * __half2float(D[out_m * ldd + out_n]);
                }
                if (j + 1 < BLOCK_N && out_n + 1 < N) {
                    val1 += beta * __half2float(D[out_m * ldd + out_n + 1]);
                }
            }
            
            if (out_n < N) {
                D[out_m * ldd + out_n] = __float2half(val0);
            }
            if (j + 1 < BLOCK_N && out_n + 1 < N) {
                D[out_m * ldd + out_n + 1] = __float2half(val1);
            }
        }
    }
}

// Simpler kernel for small sizes or as fallback
__global__ void gemm_permute_simple_kernel(
    int M, int N, int K, float alpha,
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    float beta,
    __half* __restrict__ D, int ldd)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += __half2float(A[row * lda + k]) * __half2float(B[k * ldb + col]);
    }
    
    sum *= alpha;
    if (beta != 0.0f) {
        sum += beta * __half2float(D[row * ldd + col]);
    }
    
    D[row * ldd + col] = __float2half(sum);
}

cudaError_t GemmPermute(
    int M, int N, int K, float alpha,
    __half const *A, int lda,
    __half const *B, int ldb,
    float beta,
    __half *D, int ldd)
{
    cudaError_t err = cudaSuccess;
    
    // Choose kernel based on problem size
    // For small problems, use simple kernel
    if (M <= 128 || N <= 128 || K <= 128) {
        const int BLOCK_SIZE = 16;
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        gemm_permute_simple_kernel<<<grid, block>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
    } else {
        // Use tiled kernel for larger problems
        dim3 block(THREADS_PER_BLOCK);
        dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
        
        gemm_permute_kernel<<<grid, block>>>(
            M, N, K, alpha, A, lda, B, ldb, beta, D, ldd);
    }
    
    err = cudaGetLastError();
    return err;
}
