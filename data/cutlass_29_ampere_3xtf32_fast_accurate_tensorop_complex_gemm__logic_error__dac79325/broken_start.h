#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Complex multiplication: (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
// Using 3xTF32 approach for higher accuracy on Ampere
// We split each float into high and low parts for accumulation

__device__ __forceinline__ float mul_add(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}

// Split float into high (26 bits) and low parts for 3xTF32
__device__ __forceinline__ void split_float(float x, float &hi, float &lo) {
    // Split into 26-bit high part and remaining low part
    // This is similar to TF32 which has 10-bit mantissa
    // For 3xTF32, we use 26-bit high part
    const float scale = 1.0f + (1 << 12);  // 2^12 + 1 = 4097
    float t = x * scale;
    hi = t - (t - x);
    lo = x - hi;
}

// Accurate complex multiplication using 3xTF32 approach
// (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
__device__ __forceinline__ void complex_mul_3xtf32(float ar, float ai, float br, float bi,
                                                   float &cr, float &ci) {
    // Split inputs for higher precision
    float ar_hi, ar_lo, ai_hi, ai_lo;
    float br_hi, br_lo, bi_hi, bi_lo;
    
    split_float(ar, ar_hi, ar_lo);
    split_float(ai, ai_hi, ai_lo);
    split_float(br, br_hi, br_lo);
    split_float(bi, bi_hi, bi_lo);
    
    // Compute products with error compensation
    // ac = ar*br
    float ac_hi = ar_hi * br_hi;
    float ac_lo = ar_lo * br_hi + ar_hi * br_lo + ar_lo * br_lo;
    float ac = ac_hi + ac_lo;
    
    // bd = ai*bi  
    float bd_hi = ai_hi * bi_hi;
    float bd_lo = ai_lo * bi_hi + ai_hi * bi_lo + ai_lo * bi_lo;
    float bd = bd_hi + bd_lo;
    
    // ad = ar*bi
    float ad_hi = ar_hi * bi_hi;
    float ad_lo = ar_lo * bi_hi + ar_hi * bi_lo + ar_lo * bi_lo;
    float ad = ad_hi + ad_lo;
    
    // bc = ai*br
    float bc_hi = ai_hi * br_hi;
    float bc_lo = ai_lo * br_hi + ai_hi * br_lo + ai_lo * br_lo;
    float bc = bc_hi + bc_lo;
    
    cr = ac - bd;
    ci = ad + bc;
}

// Complex multiplication with accumulation
__device__ __forceinline__ void complex_mul_add_3xtf32(float ar, float ai, float br, float bi,
                                                       float &acc_r, float &acc_i) {
    float pr, pi;
    complex_mul_3xtf32(ar, ai, br, bi, pr, pi);
    acc_r += pr;
    acc_i += pi;
}

// Main kernel for complex GEMM
// D = alpha * A * B + beta * C
// A: MxK column-major, B: KxN row-major, C/D: MxN row-major
template<int BM=128, int BN=128, int BK=8, int WM=64, int WN=64, int WNITER=2>
__global__ void complex_gemm_kernel(
    int M, int N, int K,
    float alpha_real, float alpha_imag,
    float const * __restrict__ A, int lda,
    float const * __restrict__ B, int ldb,
    float beta_real, float beta_imag,
    float const * __restrict__ C, int ldc,
    float * __restrict__ D, int ldd) {
    
    // Each block processes BM x BN output elements
    // Each warp processes WM x WN elements
    
    const int warp_size = 32;
    const int warps_per_block_x = BN / WN;
    const int warps_per_block_y = BM / WM;
    const int num_warps = warps_per_block_x * warps_per_block_y;
    
    const int warp_id = threadIdx.x / warp_size;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_row = warp_id / warps_per_block_x;
    const int warp_col = warp_id % warps_per_block_x;
    
    // Block position
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    
    // Shared memory for A and B tiles
    // A tile: BM x BK complex elements -> 2*BM*BK floats
    // B tile: BK x BN complex elements -> 2*BK*BN floats
    __shared__ float smem_A[2 * BM * BK];
    __shared__ float smem_B[2 * BK * BN];
    
    // Registers for accumulation (per thread)
    // Each warp computes WM x WN, distributed among threads
    // Using 8x8 tile per warp, each thread handles multiple elements
    
    const int thread_m = 8;  // elements per thread in M
    const int thread_n = 4;  // elements per thread in N
    
    float acc_r[thread_m][thread_n];
    float acc_i[thread_m][thread_n];
    
    #pragma unroll
    for (int i = 0; i < thread_m; i++) {
        #pragma unroll
        for (int j = 0; j < thread_n; j++) {
            acc_r[i][j] = 0.0f;
            acc_i[i][j] = 0.0f;
        }
    }
    
    // Thread mapping within warp for output
    // 32 threads -> 8x4 tile elements, each thread computes 1 element
    const int thread_row = lane_id / 4;
    const int thread_col = lane_id % 4;
    
    // Global thread coordinates for output
    const int global_m_start = block_row + warp_row * WM;
    const int global_n_start = block_col + warp_col * WN;
    
    // Load C and apply beta (if needed)
    float c_r[thread_m][thread_n];
    float c_i[thread_m][thread_n];
    bool load_c = (beta_real != 0.0f || beta_imag != 0.0f);
    
    // Main loop over K
    for (int k = 0; k < K; k += BK) {
        // Load A tile: BM x BK from column-major storage
        // Each thread loads multiple elements
        #pragma unroll
        for (int t = threadIdx.x; t < BM * BK; t += blockDim.x) {
            int local_row = t / BK;
            int local_col = t % BK;
            int global_row = block_row + local_row;
            int global_k = k + local_col;
            
            if (global_row < M && global_k < K) {
                // A is column-major: A[(i + k*lda)*2]
                int idx = (global_row + global_k * lda) * 2;
                smem_A[2 * t] = A[idx];      // real
                smem_A[2 * t + 1] = A[idx + 1]; // imag
            } else {
                smem_A[2 * t] = 0.0f;
                smem_A[2 * t + 1] = 0.0f;
            }
        }
        
        // Load B tile: BK x BN from row-major storage
        #pragma unroll
        for (int t = threadIdx.x; t < BK * BN; t += blockDim.x) {
            int local_row = t / BN;
            int local_col = t % BN;
            int global_k = k + local_row;
            int global_col = block_col + local_col;
            
            if (global_k < K && global_col < N) {
                // B is row-major: B[(k*ldb + j)*2]
                int idx = (global_k * ldb + global_col) * 2;
                smem_B[2 * t] = B[idx];      // real
                smem_B[2 * t + 1] = B[idx + 1]; // imag
            } else {
                smem_B[2 * t] = 0.0f;
                smem_B[2 * t + 1] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute on tile
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load A elements for this thread
            float a_r[thread_m];
            float a_i[thread_m];
            
            #pragma unroll
            for (int i = 0; i < thread_m; i++) {
                int local_m = warp_row * WM + thread_row + i * 8;  // stride by 8
                int idx = 2 * (local_m * BK + kk);
                a_r[i] = smem_A[idx];
                a_i[i] = smem_A[idx + 1];
            }
            
            // Load B elements for this thread
            float b_r[thread_n];
            float b_i[thread_n];
            
            #pragma unroll
            for (int j = 0; j < thread_n; j++) {
                int local_n = warp_col * WN + thread_col + j * 4;  // stride by 4
                int idx = 2 * (kk * BN + local_n);
                b_r[j] = smem_B[idx];
                b_i[j] = smem_B[idx + 1];
            }
            
            // Multiply-accumulate
            #pragma unroll
            for (int i = 0; i < thread_m; i++) {
                #pragma unroll
                for (int j = 0; j < thread_n; j++) {
                    complex_mul_add_3xtf32(a_r[i], a_i[i], b_r[j], b_i[j], 
                                          acc_r[i][j], acc_i[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Apply alpha and beta, write output
    // Load C if beta is non-zero
    if (load_c) {
        #pragma unroll
        for (int i = 0; i < thread_m; i++) {
            int global_m = global_m_start + thread_row + i * 8;
            #pragma unroll
            for (int j = 0; j < thread_n; j++) {
                int global_n = global_n_start + thread_col + j * 4;
                
                if (global_m < M && global_n < N) {
                    int idx = (global_m * ldc + global_n) * 2;
                    c_r[i][j] = C[idx];
                    c_i[i][j] = C[idx + 1];
                }
            }
        }
    }
    
    // Compute final result: alpha * acc + beta * c
    #pragma unroll
    for (int i = 0; i < thread_m; i++) {
        int global_m = global_m_start + thread_row + i * 8;
        #pragma unroll
        for (int j = 0; j < thread_n; j++) {
            int global_n = global_n_start + thread_col + j * 4;
            
            if (global_m < M && global_n < N) {
                // alpha * acc
                float alpha_acc_r, alpha_acc_i;
                complex_mul_3xtf32(alpha_real, alpha_imag, 
                                  acc_r[i][j], acc_i[i][j],
                                  alpha_acc_r, alpha_acc_i);
                
                // beta * c
                float beta_c_r = 0.0f, beta_c_i = 0.0f;
                if (load_c) {
                    complex_mul_3xtf32(beta_real, beta_imag,
                                      c_r[i][j], c_i[i][j],
                                      beta_c_r, beta_c_i);
                }
                
                // Sum
                float d_r = alpha_acc_r + beta_c_r;
                float d_i = alpha_acc_i + beta_c_i;
                
                int idx = (global_m * ldd + global_n) * 2;
                D[idx] = d_r;
                D[idx + 1] = d_i;
            }
        }
    }
}

// Simpler kernel for small sizes
__global__ void complex_gemm_small_kernel(
    int M, int N, int K,
    float alpha_real, float alpha_imag,
    float const * __restrict__ A, int lda,
    float const * __restrict__ B, int ldb,
    float beta_real, float beta_imag,
    float const * __restrict__ C, int ldc,
    float * __restrict__ D, int ldd) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float acc_r = 0.0f;
    float acc_i = 0.0f;
    
    for (int k = 0; k < K; k++) {
        // A[row, k] - column major
        int a_idx = (row + k * lda) * 2;
        float ar = A[a_idx];
        float ai = A[a_idx + 1];
        
        // B[k, col] - row major
        int b_idx = (k * ldb + col) * 2;
        float br = B[b_idx];
        float bi = B[b_idx + 1];
        
        // Complex multiply with 3xTF32
        complex_mul_add_3xtf32(ar, ai, br, bi, acc_r, acc_i);
    }
    
    // Apply alpha
    float alpha_acc_r, alpha_acc_i;
    complex_mul_3xtf32(alpha_real, alpha_imag, acc_r, acc_i, alpha_acc_r, alpha_acc_i);
    
    // Apply beta and add
    float d_r = alpha_acc_r;
    float d_i = alpha_acc_i;
    
    if (beta_real != 0.0f || beta_imag != 0.0f) {
        int c_idx = (row * ldc + col) * 2;
        float cr = C[c_idx];
        float ci = C[c_idx + 1];
        
        float beta_c_r, beta_c_i;
        complex_mul_3xtf32(beta_real, beta_imag, cr, ci, beta_c_r, beta_c_i);
        
        d_r += beta_c_r;
        d_i += beta_c_i;
    }
    
    int d_idx = (row * ldd + col) * 2;
    D[d_idx] = d_r;
    D[d_idx + 1] = d_i;
}

// Optimized kernel using Tensor Core-like approach with warp-level operations
template<int BM=128, int BN=128, int BK=16>
__global__ void complex_gemm_optimized_kernel(
    int M, int N, int K,
    float alpha_real, float alpha_imag,
    float const * __restrict__ A, int lda,
    float const * __restrict__ B, int ldb,
    float beta_real, float beta_imag,
    float const * __restrict__ C, int ldc,
    float * __restrict__ D, int ldd) {
    
    const int warp_size = 32;
    const int num_warps = (blockDim.x + warp_size - 1) / warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int lane_id = threadIdx.x % warp_size;
    
    // 4 warps per block, each warp handles 64x64 tile
    const int warp_m = 64;
    const int warp_n = 64;
    const int warps_m = 2;
    const int warps_n = 2;
    
    const int warp_row = warp_id / warps_n;
    const int warp_col = warp_id % warps_n;
    
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    
    // Shared memory
    __shared__ float smem_A[2 * BM * BK];
    __shared__ float smem_B[2 * BK * BN];
    
    // Accumulators - each thread handles 4x4 complex elements
    float acc_r[4][4];
    float acc_i[4][4];
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            acc_r[i][j] = 0.0f;
            acc_i[i][j] = 0.0f;
        }
    }
    
    // Thread layout within warp: 8x4 threads, each handles 8x16 elements
    const int threads_m = 8;
    const int threads_n = 4;
    const int thread_row = lane_id / threads_n;
    const int thread_col = lane_id % threads_n;
    
    // Global positions
    const int global_m_base = block_row + warp_row * warp_m;
    const int global_n_base = block_col + warp_col * warp_n;
    
    // Main loop
    for (int k = 0; k < K; k += BK) {
        // Load A: BM x BK
        #pragma unroll
        for (int t = threadIdx.x; t < BM * BK; t += blockDim.x) {
            int local_m = t / BK;
            int local_k = t % BK;
            int global_m = block_row + local_m;
            int global_k = k + local_k;
            
            float ar = 0.0f, ai = 0.0f;
            if (global_m < M && global_k < K) {
                int idx = (global_m + global_k * lda) * 2;
                ar = A[idx];
                ai = A[idx + 1];
            }
            smem_A[2 * t] = ar;
            smem_A[2 * t + 1] = ai;
        }
        
        // Load B: BK x BN
        #pragma unroll
        for (int t = threadIdx.x; t < BK * BN; t += blockDim.x) {
            int local_k = t / BN;
            int local_n = t % BN;
            int global_k = k + local_k;
            int global_n = block_col + local_n;
            
            float br = 0.0f, bi = 0.0f;
            if (global_k < K && global_n < N) {
                int idx = (global_k * ldb + global_n) * 2;
                br = B[idx];
                bi = B[idx + 1];
            }
            smem_B[2 * t] = br;
            smem_B[2 * t + 1] = bi;
        }
        
        __syncthreads();
        
        // Compute tile
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load A for this warp
            float a_r[4], a_i[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = warp_row * warp_m + thread_row * 8 + i * 2;
                int idx = 2 * (m * BK + kk);
                a_r[i] = smem_A[idx];
                a_i[i] = smem_A[idx + 1];
            }
            
            // Load B for this warp
            float b_r[4], b_i[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int n = warp_col * warp_n + thread_col * 16 + j * 4;
                int idx = 2 * (kk * BN + n);
                b_r[j] = smem_B[idx];
                b_i[j] = smem_B[idx + 1];
            }
            
            // MAC
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    complex_mul_add_3xtf32(a_r[i], a_i[i], b_r[j], b_i[j],
                                          acc_r[i][j], acc_i[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int global_m = global_m_base + thread_row * 8 + i * 2;
        for (int j = 0; j < 4; j++) {
            int global_n = global_n_base + thread_col * 16 + j * 4;
            
            if (global_m < M && global_n < N) {
                // Apply alpha
                float val_r, val_i;
                complex_mul_3xtf32(alpha_real, alpha_imag, 
                                  acc_r[i][j], acc_i[i][j],
                                  val_r, val_i);
                
                // Apply beta * C
                if (beta_real != 0.0f || beta_imag != 0.0f) {
                    int c_idx = (global_m * ldc + global_n) * 2;
                    float cr = C[c_idx];
                    float ci = C[c_idx + 1];
                    
                    float beta_cr, beta_ci;
                    complex_mul_3xtf32(beta_real, beta_imag, cr, ci, beta_cr, beta_ci);
                    
                    val_r += beta_cr;
                    val_i += beta_ci;
                }
                
                int d_idx = (global_m * ldd + global_n) * 2;
                D[d_idx] = val_r;
                D[d_idx + 1] = val_i;
            }
        }
    }
}

// Host wrapper
cudaError_t ComplexGemm(
    int M, int N, int K,
    float alpha_real, float alpha_imag,
    float const *A, int lda,
    float const *B, int ldb,
    float beta_real, float beta_imag,
    float const *C, int ldc,
    float *D, int ldd) {
    
    // Choose kernel based on size
    if (M <= 64 || N <= 64 || K <= 64) {
        // Small kernel
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        complex_gemm_small_kernel<<<grid, block>>>(
            M, N, K, alpha_real, alpha_imag, A, lda, B, ldb,
            beta_real, beta_imag, C, ldc, D, ldd);
    } else {
        // Optimized kernel for larger sizes
        const int BM = 128;
        const int BN = 128;
        const int BK = 16;
        
        dim3 block(128);  // 4 warps
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
        
        complex_gemm_optimized_kernel<BM, BN, BK><<<grid, block>>>(
            M, N, K, alpha_real, alpha_imag, A, lda, B, ldb,
            beta_real, beta_imag, C, ldc, D, ldd);
    }
    
    return cudaGetLastError();
}
