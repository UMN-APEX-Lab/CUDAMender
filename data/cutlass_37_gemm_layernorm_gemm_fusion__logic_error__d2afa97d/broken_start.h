#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

// Helper to convert half to float
__device__ __forceinline__ float half2float(__half h) {
    return __half2float(h);
}

// GEMM0: C0 = alpha * A0 * B0 + beta * C0_in
// A0: row-major [N0, K0], stride lda0
// B0: column-major [K0, M], stride ldb0
// C0: column-major [N0, M], stride N0

// LayerNorm: per-column normalization
// For each column c of C0 (shape N0 x M):
//   mean = avg(C0[:,c])
//   var = avg((C0[:,c] - mean)^2)
//   inv_std = 1/sqrt(var + 1e-6)
//   normalized[i,c] = (C0[i,c] - mean) * inv_std * gamma[i] + beta_ln[i]

// GEMM1: C1 = alpha * A1 * normalized
// A1: column-major [N1, N0], stride lda1
// normalized: column-major [N0, M]
// C1: column-major [N1, M], stride ldc1

// Block size configuration
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 32;
constexpr int BLOCK_K = 16;

// Warp size
constexpr int WARP_SIZE = 32;

// First kernel: GEMM0 + compute mean/var for LayerNorm
// We'll compute GEMM0 and store in shared memory, then compute mean/var per column
template <int BM, int BN, int BK>
__global__ void gemm0_layernorm_part1_kernel(
    int M, int N0, int K0,
    float alpha, float beta,
    const __half* A0, int lda0,
    const __half* B0, int ldb0,
    __half* C0,  // intermediate output [N0, M] column-major
    float* mean, // [M] mean per column
    float* inv_std // [M] inv_std per column
) {
    // Block indices
    int bx = blockIdx.x; // M dimension
    int by = blockIdx.y; // N0 dimension
    
    // Thread indices
    int tx = threadIdx.x % 8;
    int ty = threadIdx.x / 8;
    
    // Starting positions
    int m_start = bx * BM;
    int n_start = by * BN;
    
    // Shared memory for A and B tiles
    __shared__ __half sA[BK][BM];  // A tile: row-major in shared
    __shared__ __half sB[BK][BN];  // B tile: row-major in shared
    
    // Accumulator registers
    float acc[BM/8][BN/4];  // Each thread computes BM/8 x BN/4 elements
    
    #pragma unroll
    for (int i = 0; i < BM/8; i++) {
        #pragma unroll
        for (int j = 0; j < BN/4; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Main loop over K
    for (int k = 0; k < K0; k += BK) {
        // Load A tile: A0 is row-major [N0, K0]
        // Each thread loads elements
        #pragma unroll
        for (int i = 0; i < BK; i += 4) {
            int row = n_start + ty * 4 + (threadIdx.x / 8);
            int col = k + (threadIdx.x % 8) * 4 + i;
            if (row < N0 && col < K0) {
                sA[i + (threadIdx.x % 8)][row - n_start] = A0[row * lda0 + col];
            } else {
                sA[i + (threadIdx.x % 8)][row - n_start] = __float2half(0.0f);
            }
        }
        
        // Load B tile: B0 is column-major [K0, M]
        #pragma unroll
        for (int i = 0; i < BK; i += 4) {
            int row = k + ty * 4 + (threadIdx.x / 8);
            int col = m_start + (threadIdx.x % 8) * 4 + i;
            if (row < K0 && col < M) {
                sB[i + (threadIdx.x % 8)][row - k] = B0[row + col * ldb0];
            } else {
                sB[i + (threadIdx.x % 8)][row - k] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // Compute partial dot products
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            #pragma unroll
            for (int i = 0; i < BM/8; i++) {
                #pragma unroll
                for (int j = 0; j < BN/4; j++) {
                    float a_val = half2float(sA[kk][ty * (BM/8) + i]);
                    float b_val = half2float(sB[kk][tx * (BN/4) + j]);
                    acc[i][j] += a_val * b_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Apply alpha and write to C0 (column-major)
    #pragma unroll
    for (int i = 0; i < BM/8; i++) {
        #pragma unroll
        for (int j = 0; j < BN/4; j++) {
            int m_idx = m_start + ty * (BM/8) + i;
            int n_idx = n_start + tx * (BN/4) + j;
            if (m_idx < M && n_idx < N0) {
                float val = alpha * acc[i][j];  // beta * C0_in not needed as C0 is intermediate
                C0[n_idx + m_idx * N0] = __float2half(val);
            }
        }
    }
    
    // Compute mean and variance using warp-level reduction
    // Each warp processes one column (m_idx)
    // We need to compute mean of C0[:, m_idx] and variance
    
    // Simpler approach: use shared memory for reduction
    // Each thread in a warp handles partial sum
    
    __shared__ float s_mean[BM];
    __shared__ float s_var[BM];
    
    // Initialize
    if (threadIdx.x < BM) {
        s_mean[threadIdx.x] = 0.0f;
        s_var[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    // Each thread computes partial sum for its assigned m indices
    #pragma unroll
    for (int i = 0; i < BM/8; i++) {
        int m_idx = m_start + ty * (BM/8) + i;
        if (m_idx < M) {
            float thread_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < BN/4; j++) {
                thread_sum += acc[i][j] * alpha;
            }
            atomicAdd(&s_mean[ty * (BM/8) + i], thread_sum);
        }
    }
    __syncthreads();
    
    // Compute mean
    if (threadIdx.x < BM && m_start + threadIdx.x < M) {
        s_mean[threadIdx.x] /= N0;
        mean[m_start + threadIdx.x] = s_mean[threadIdx.x];
    }
    __syncthreads();
    
    // Compute variance
    #pragma unroll
    for (int i = 0; i < BM/8; i++) {
        int m_idx = m_start + ty * (BM/8) + i;
        if (m_idx < M) {
            float col_mean = s_mean[ty * (BM/8) + i];
            float thread_var_sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < BN/4; j++) {
                int n_idx = n_start + tx * (BN/4) + j;
                if (n_idx < N0) {
                    float diff = acc[i][j] * alpha - col_mean;
                    thread_var_sum += diff * diff;
                }
            }
            atomicAdd(&s_var[ty * (BM/8) + i], thread_var_sum);
        }
    }
    __syncthreads();
    
    // Compute inv_std
    if (threadIdx.x < BM && m_start + threadIdx.x < M) {
        float variance = s_var[threadIdx.x] / N0;
        inv_std[m_start + threadIdx.x] = 1.0f / sqrtf(variance + 1e-6f);
    }
}

// Second kernel: Apply LayerNorm normalization and GEMM1
// normalized[i,c] = (C0[i,c] - mean[c]) * inv_std[c] * gamma[i] + beta_ln[i]
// Then C1 = A1 * normalized

template <int BM, int BN, int BK>
__global__ void layernorm_gemm1_kernel(
    int M, int N0, int N1,
    float alpha,
    const __half* C0,      // [N0, M] column-major
    const float* mean,     // [M]
    const float* inv_std,  // [M]
    const __half* gamma,   // [N0]
    const __half* beta_ln, // [N0]
    const __half* A1, int lda1,  // [N1, N0] column-major
    __half* C1, int ldc1   // [N1, M] column-major
) {
    // Block indices
    int bx = blockIdx.x; // M dimension
    int by = blockIdx.y; // N1 dimension
    
    // Thread indices
    int tx = threadIdx.x % 8;
    int ty = threadIdx.x / 8;
    
    int m_start = bx * BM;
    int n1_start = by * BN;
    
    // Shared memory for normalized values and A1 tile
    __shared__ __half sNorm[BK][BM];  // normalized tile
    __shared__ __half sA1[BK][BN];    // A1 tile
    
    // Load mean and inv_std for this column block
    __shared__ float s_mean[BM];
    __shared__ float s_inv_std[BM];
    
    if (threadIdx.x < BM && m_start + threadIdx.x < M) {
        s_mean[threadIdx.x] = mean[m_start + threadIdx.x];
        s_inv_std[threadIdx.x] = inv_std[m_start + threadIdx.x];
    }
    __syncthreads();
    
    // Accumulator
    float acc[BM/8][BN/4];
    #pragma unroll
    for (int i = 0; i < BM/8; i++) {
        #pragma unroll
        for (int j = 0; j < BN/4; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Main loop over N0
    for (int n0 = 0; n0 < N0; n0 += BK) {
        // Load normalized values: apply LayerNorm on the fly
        // normalized[i,c] = (C0[i,c] - mean[c]) * inv_std[c] * gamma[i] + beta_ln[i]
        #pragma unroll
        for (int i = 0; i < BK; i += 4) {
            int n0_idx = n0 + i + (threadIdx.x / 8);
            int m_idx = m_start + (threadIdx.x % 8) * 4;
            
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (n0_idx < N0 && m_idx + j < M) {
                    float c0_val = half2float(C0[n0_idx + (m_idx + j) * N0]);
                    float norm_val = (c0_val - s_mean[m_idx + j - m_start]) * s_inv_std[m_idx + j - m_start];
                    float g = half2float(gamma[n0_idx]);
                    float b = half2float(beta_ln[n0_idx]);
                    float normalized = norm_val * g + b;
                    sNorm[i + (threadIdx.x / 8)][(threadIdx.x % 8) * 4 + j] = __float2half(normalized);
                } else {
                    sNorm[i + (threadIdx.x / 8)][(threadIdx.x % 8) * 4 + j] = __float2half(0.0f);
                }
            }
        }
        
        // Load A1 tile: A1 is column-major [N1, N0]
        #pragma unroll
        for (int i = 0; i < BK; i += 4) {
            int row = n0 + i + (threadIdx.x / 8);
            int col = n1_start + (threadIdx.x % 8) * 4;
            
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                if (row < N0 && col + j < N1) {
                    sA1[i + (threadIdx.x / 8)][(threadIdx.x % 8) * 4 + j] = A1[row + (col + j) * lda1];
                } else {
                    sA1[i + (threadIdx.x / 8)][(threadIdx.x % 8) * 4 + j] = __float2half(0.0f);
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial dot products
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            #pragma unroll
            for (int i = 0; i < BM/8; i++) {
                #pragma unroll
                for (int j = 0; j < BN/4; j++) {
                    float norm_val = half2float(sNorm[kk][ty * (BM/8) + i]);
                    float a1_val = half2float(sA1[kk][tx * (BN/4) + j]);
                    acc[i][j] += norm_val * a1_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    #pragma unroll
    for (int i = 0; i < BM/8; i++) {
        #pragma unroll
        for (int j = 0; j < BN/4; j++) {
            int m_idx = m_start + ty * (BM/8) + i;
            int n1_idx = n1_start + tx * (BN/4) + j;
            if (m_idx < M && n1_idx < N1) {
                C1[n1_idx + m_idx * ldc1] = __float2half(alpha * acc[i][j]);
            }
        }
    }
}

// Simpler fused kernel: process one column of M at a time
// Each block handles one column of M, all threads in block compute GEMM0, LayerNorm, GEMM1 for that column
__global__ void fused_gemm_layernorm_gemm_kernel(
    int M, int N0, int K0, int N1,
    float alpha, float beta,
    const __half* A0, int lda0,
    const __half* B0, int ldb0,
    const __half* A1, int lda1,
    __half* C1, int ldc1,
    const __half* gamma,
    const __half* beta_ln
) {
    // Each block handles one column of M
    int m_col = blockIdx.x;
    if (m_col >= M) return;
    
    // Shared memory for C0 column and intermediate results
    extern __shared__ __half shared_mem[];
    __half* s_C0 = shared_mem;           // [N0] - C0 column
    float* s_normalized = (float*)(s_C0 + ((N0 + 31) & ~31)); // [N0] - normalized as float
    
    int tid = threadIdx.x;
    int nt = blockDim.x;
    
    // Step 1: GEMM0 - compute C0[:, m_col] = alpha * A0 * B0[:, m_col]
    // A0: row-major [N0, K0], B0: column-major [K0, M]
    
    for (int n = tid; n < N0; n += nt) {
        float sum = 0.0f;
        for (int k = 0; k < K0; k++) {
            float a = half2float(A0[n * lda0 + k]);
            float b = half2float(B0[k + m_col * ldb0]);
            sum += a * b;
        }
        s_C0[n] = __float2half(alpha * sum);
    }
    __syncthreads();
    
    // Step 2: LayerNorm - compute mean and variance
    // Use parallel reduction
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    
    // Compute mean
    float local_sum = 0.0f;
    for (int n = tid; n < N0; n += nt) {
        local_sum += half2float(s_C0[n]);
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    if (tid == 0) {
        s_mean = local_sum / N0;
    }
    __syncthreads();
    
    // Compute variance
    float local_var = 0.0f;
    float mean_val = s_mean;
    for (int n = tid; n < N0; n += nt) {
        float diff = half2float(s_C0[n]) - mean_val;
        local_var += diff * diff;
    }
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_var += __shfl_down_sync(0xffffffff, local_var, offset);
    }
    
    if (tid == 0) {
        float variance = local_var / N0;
        s_inv_std = 1.0f / sqrtf(variance + 1e-6f);
    }
    __syncthreads();
    
    // Apply LayerNorm
    float inv_std_val = s_inv_std;
    for (int n = tid; n < N0; n += nt) {
        float normalized = (half2float(s_C0[n]) - mean_val) * inv_std_val;
        float g = half2float(gamma[n]);
        float b = half2float(beta_ln[n]);
        s_normalized[n] = normalized * g + b;
    }
    __syncthreads();
    
    // Step 3: GEMM1 - compute C1[:, m_col] = A1 * normalized
    // A1: column-major [N1, N0]
    
    for (int n1 = tid; n1 < N1; n1 += nt) {
        float sum = 0.0f;
        for (int n0 = 0; n0 < N0; n0++) {
            float a = half2float(A1[n0 + n1 * lda1]);
            sum += a * s_normalized[n0];
        }
        C1[n1 + m_col * ldc1] = __float2half(alpha * sum);
    }
}

// Optimized version with better memory access patterns
__global__ void fused_gemm_layernorm_gemm_kernel_v2(
    int M, int N0, int K0, int N1,
    float alpha, float beta,
    const __half* A0, int lda0,
    const __half* B0, int ldb0,
    const __half* A1, int lda1,
    __half* C1, int ldc1,
    const __half* gamma,
    const __half* beta_ln
) {
    // Each block handles multiple columns of M to improve occupancy
    // Process BM columns per block
    const int BM = 4;  // Columns of M per block
    
    int m_base = blockIdx.x * BM;
    int tid = threadIdx.x;
    int nt = blockDim.x;
    
    extern __shared__ __half shared_mem[];
    // Layout: C0[BM][N0], normalized[BM][N0], mean[BM], inv_std[BM]
    __half* s_C0 = shared_mem;
    float* s_normalized = (float*)(s_C0 + BM * ((N0 + 15) & ~15));
    float* s_mean = s_normalized + BM * ((N0 + 15) & ~15);
    float* s_inv_std = s_mean + BM;
    
    // Process each column in the block
    for (int bm = 0; bm < BM && m_base + bm < M; bm++) {
        int m_col = m_base + bm;
        
        // GEMM0 for this column
        for (int n = tid; n < N0; n += nt) {
            float sum = 0.0f;
            // Unroll by 4 for better instruction throughput
            int k = 0;
            for (; k + 3 < K0; k += 4) {
                float a0 = half2float(A0[n * lda0 + k]);
                float a1 = half2float(A0[n * lda0 + k + 1]);
                float a2 = half2float(A0[n * lda0 + k + 2]);
                float a3 = half2float(A0[n * lda0 + k + 3]);
                
                float b0 = half2float(B0[k + m_col * ldb0]);
                float b1 = half2float(B0[k + 1 + m_col * ldb0]);
                float b2 = half2float(B0[k + 2 + m_col * ldb0]);
                float b3 = half2float(B0[k + 3 + m_col * ldb0]);
                
                sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            }
            for (; k < K0; k++) {
                sum += half2float(A0[n * lda0 + k]) * half2float(B0[k + m_col * ldb0]);
            }
            s_C0[bm * N0 + n] = __float2half(alpha * sum);
        }
    }
    __syncthreads();
    
    // LayerNorm for each column
    for (int bm = 0; bm < BM && m_base + bm < M; bm++) {
        // Compute mean
        float local_sum = 0.0f;
        for (int n = tid; n < N0; n += nt) {
            local_sum += half2float(s_C0[bm * N0 + n]);
        }
        
        // Block reduction
        __shared__ float s_scratch[256];
        s_scratch[tid] = local_sum;
        __syncthreads();
        
        for (int stride = nt / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_scratch[tid] += s_scratch[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            s_mean[bm] = s_scratch[0] / N0;
        }
        __syncthreads();
        
        // Compute variance
        float mean_val = s_mean[bm];
        float local_var = 0.0f;
        for (int n = tid; n < N0; n += nt) {
            float diff = half2float(s_C0[bm * N0 + n]) - mean_val;
            local_var += diff * diff;
        }
        
        s_scratch[tid] = local_var;
        __syncthreads();
        
        for (int stride = nt / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_scratch[tid] += s_scratch[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            float variance = s_scratch[0] / N0;
            s_inv_std[bm] = 1.0f / sqrtf(variance + 1e-6f);
        }
        __syncthreads();
        
        // Apply LayerNorm
        float inv_std_val = s_inv_std[bm];
        for (int n = tid; n < N0; n += nt) {
            float normalized = (half2float(s_C0[bm * N0 + n]) - mean_val) * inv_std_val;
            float g = half2float(gamma[n]);
            float b = half2float(beta_ln[n]);
            s_normalized[bm * N0 + n] = normalized * g + b;
        }
    }
    __syncthreads();
    
    // GEMM1 for each column
    for (int bm = 0; bm < BM && m_base + bm < M; bm++) {
        int m_col = m_base + bm;
        
        for (int n1 = tid; n1 < N1; n1 += nt) {
            float sum = 0.0f;
            // Unroll by 4
            int n0 = 0;
            for (; n0 + 3 < N0; n0 += 4) {
                float a0 = half2float(A1[n0 + n1 * lda1]);
                float a1 = half2float(A1[n0 + 1 + n1 * lda1]);
                float a2 = half2float(A1[n0 + 2 + n1 * lda1]);
                float a3 = half2float(A1[n0 + 3 + n1 * lda1]);
                
                float n0_val = s_normalized[bm * N0 + n0];
                float n1_val = s_normalized[bm * N0 + n0 + 1];
                float n2_val = s_normalized[bm * N0 + n0 + 2];
                float n3_val = s_normalized[bm * N0 + n0 + 3];
                
                sum += a0 * n0_val + a1 * n1_val + a2 * n2_val + a3 * n3_val;
            }
            for (; n0 < N0; n0++) {
                sum += half2float(A1[n0 + n1 * lda1]) * s_normalized[bm * N0 + n0];
            }
            C1[n1 + m_col * ldc1] = __float2half(alpha * sum);
        }
    }
}

// Main solution function
cudaError_t gemm_layernorm_gemm_solution(
    int M_sol, int N0_sol, int K0_sol, int N1_sol,
    float alpha, float beta,
    __half const *A0, int lda0,
    __half const *B0, int ldb0,
    __half const *A1, int lda1,
    __half *C1, int ldc1,
    __half const *gamma,
    __half const *beta_ln,
    __half const *shifted_K,
    bool is_column_major
) {
    // Only support column-major mode as specified
    if (!is_column_major) {
        // Could add row-major support, but task specifies column-major
        return cudaErrorNotSupported;
    }
    
    // Use the fused kernel with one column per block for simplicity and correctness
    // Each block handles BM columns to improve occupancy
    
    const int BM = 2;  // Columns per block
    
    int num_blocks = (M_sol + BM - 1) / BM;
    int threads_per_block = 256;
    
    // Calculate shared memory size
    // C0[BM][N0], normalized[BM][N0], mean[BM], inv_std[BM]
    size_t shared_mem_size = BM * N0_sol * sizeof(__half) +    // s_C0
                             BM * N0_sol * sizeof(float) +     // s_normalized  
                             BM * sizeof(float) +               // s_mean
                             BM * sizeof(float);                // s_inv_std
    
    // Round up to avoid alignment issues
    shared_mem_size = (shared_mem_size + 255) & ~255;
    
    // Limit shared memory to 48KB
    if (shared_mem_size > 48 * 1024) {
        // Fall back to single column per block
        const int BM_FALLBACK = 1;
        num_blocks = M_sol;
        shared_mem_size = BM_FALLBACK * N0_sol * sizeof(__half) +
                         BM_FALLBACK * N0_sol * sizeof(float) +
                         BM_FALLBACK * sizeof(float) +
                         BM_FALLBACK * sizeof(float);
        shared_mem_size = (shared_mem_size + 255) & ~255;
        
        fused_gemm_layernorm_gemm_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
            M_sol, N0_sol, K0_sol, N1_sol,
            alpha, beta,
            A0, lda0,
            B0, ldb0,
            A1, lda1,
            C1, ldc1,
            gamma, beta_ln
        );
    } else {
        fused_gemm_layernorm_gemm_kernel_v2<<<num_blocks, threads_per_block, shared_mem_size>>>(
            M_sol, N0_sol, K0_sol, N1_sol,
            alpha, beta,
            A0, lda0,
            B0, ldb0,
            A1, lda1,
            C1, ldc1,
            gamma, beta_ln
        );
    }
    
    return cudaGetLastError();
}
