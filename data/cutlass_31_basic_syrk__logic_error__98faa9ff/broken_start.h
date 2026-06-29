#pragma once
#include <cuda_runtime.h>

// Double-precision SYRK: C = alpha * A * A^T + beta * C
// Only lower triangle of C is computed (i >= j)
// Column-major layout

// Tile dimensions - tuned for double precision
#define SYRK_TILE_M 64
#define SYRK_TILE_N 64
#define SYRK_TILE_K 16

// Thread block size
#define SYRK_THREADS 256

__global__ void dsyrk_kernel(
    int N, int K,
    double alpha,
    const double *A, int lda,
    double beta,
    double *C, int ldc)
{
    // Each block computes a SYRK_TILE_M x SYRK_TILE_N tile of C
    // For lower triangle, we only need blocks where row_start >= col_start
    // But we launch a grid that covers the lower triangular region
    
    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;
    
    // For lower triangle, we need block_row * SYRK_TILE_M >= block_col * SYRK_TILE_N
    // Skip upper triangle blocks
    if (block_row * SYRK_TILE_M < block_col * SYRK_TILE_N) {
        return;
    }
    
    const int thread_id = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Tile boundaries
    const int row_start = block_row * SYRK_TILE_M;
    const int col_start = block_col * SYRK_TILE_N;
    
    // Shared memory for A tiles
    __shared__ double A_row_shared[SYRK_TILE_M][SYRK_TILE_K];
    __shared__ double A_col_shared[SYRK_TILE_N][SYRK_TILE_K];
    
    // Accumulator registers - each thread computes a subset of the tile
    // Use 8x8 register tiling for good performance
    const int REG_M = 8;
    const int REG_N = 8;
    
    // Number of threads needed to cover the tile with 8x8 register blocks
    // Tile is 64x64, each thread does 8x8, so we need 8x8 = 64 threads for full coverage
    // But we have 256 threads, so each thread can do multiple 8x8 blocks
    
    const int threads_per_row = SYRK_TILE_N / REG_N; // 8
    const int threads_per_col = SYRK_TILE_M / REG_M; // 8
    
    // Thread position within the tile computation
    const int thread_row = thread_id / threads_per_row;
    const int thread_col = thread_id % threads_per_row;
    
    // Each thread computes REG_M x REG_N output elements
    // But with 256 threads, we need to handle multiple positions or use fewer threads effectively
    
    // Simpler approach: use 64 threads (8x8) for the main computation, others help with loading
    // Actually let's use strided access for 256 threads
    
    double accum[REG_M][REG_N];
    #pragma unroll
    for (int i = 0; i < REG_M; i++) {
        #pragma unroll
        for (int j = 0; j < REG_N; j++) {
            accum[i][j] = 0.0;
        }
    }
    
    // Determine which 8x8 block this thread computes
    // With 256 threads, we can compute 4 such blocks (64 threads per block)
    const int sub_block = thread_id / 64;      // 0, 1, 2, or 3
    const int sub_thread = thread_id % 64;     // 0-63 within the sub-block
    
    const int sub_row = sub_block / 2;         // 0 or 1 (vertical position of 8x8 block)
    const int sub_col = sub_block % 2;         // 0 or 1 (horizontal position of 8x8 block)
    
    const int local_row = sub_thread / 8;      // 0-7 within 8x8
    const int local_col = sub_thread % 8;      // 0-7 within 8x8
    
    // Global position in tile
    const int global_row = sub_row * 32 + local_row * 4;  // stride by 4 for coalescing
    const int global_col = sub_col * 32 + local_col * 4;
    
    // Actually let's use a simpler 4x4 register tile approach that's easier to manage
    
    // Re-initialize for simpler approach
    #pragma unroll
    for (int i = 0; i < REG_M; i++) {
        #pragma unroll
        for (int j = 0; j < REG_N; j++) {
            accum[i][j] = 0.0;
        }
    }
    
    // Each thread computes a 8x8 region using 4x4 tiles for register blocking
    // Position within 64x64 tile: thread computes elements at stride
    
    const int thread_m = (thread_id % 32) * 2;  // 0, 2, 4, ... 62 (32 threads cover 64 rows, stride 2)
    const int thread_n = (thread_id / 32) * 8;  // 0, 8, 16, ... 56 (8 threads cover 64 cols, stride 8)
    
    // Main loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += SYRK_TILE_K) {
        // Load A tile for row dimension (A[row_start:row_start+SYRK_TILE_M, k_tile:k_tile+SYRK_TILE_K])
        // and A tile for column dimension (A[col_start:col_start+SYRK_TILE_N, k_tile:k_tile+SYRK_TILE_K])
        
        // Collaborative loading: each thread loads multiple elements
        const int load_per_thread = (SYRK_TILE_M * SYRK_TILE_K + num_threads - 1) / num_threads;
        for (int l = 0; l < load_per_thread; l++) {
            int idx = thread_id + l * num_threads;
            int s_row = idx / SYRK_TILE_K;
            int s_k = idx % SYRK_TILE_K;
            if (s_row < SYRK_TILE_M && k_tile + s_k < K) {
                int a_row = row_start + s_row;
                if (a_row < N) {
                    A_row_shared[s_row][s_k] = A[a_row + (k_tile + s_k) * lda];
                } else {
                    A_row_shared[s_row][s_k] = 0.0;
                }
            }
        }
        
        // Load second A tile (for column)
        const int load_per_thread2 = (SYRK_TILE_N * SYRK_TILE_K + num_threads - 1) / num_threads;
        for (int l = 0; l < load_per_thread2; l++) {
            int idx = thread_id + l * num_threads;
            int s_col = idx / SYRK_TILE_K;
            int s_k = idx % SYRK_TILE_K;
            if (s_col < SYRK_TILE_N && k_tile + s_k < K) {
                int a_col = col_start + s_col;
                if (a_col < N) {
                    A_col_shared[s_col][s_k] = A[a_col + (k_tile + s_k) * lda];
                } else {
                    A_col_shared[s_col][s_k] = 0.0;
                }
            }
        }
        
        __syncthreads();
        
        // Compute partial products for this k-tile
        // Each thread computes its assigned 8x8 region with 4x4 register blocking
        
        // Actual K for this tile
        int k_end = min(SYRK_TILE_K, K - k_tile);
        
        #pragma unroll
        for (int kk = 0; kk < SYRK_TILE_K; kk++) {
            if (kk >= k_end) break;
            
            // Load A values into registers for the rows this thread needs
            double a_rows[8];
            double a_cols[8];
            
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                int r = thread_m + (i % 2) * 4 + (i / 2);  // distribute 8 values across 2x4 pattern
                // Actually simpler: thread_m is start, need 8 consecutive or strided
                // Let's use: thread computes rows [thread_m, thread_m+1] x [thread_n, thread_n+7]
                // with 4x4 blocking
            }
            
            // Simpler: direct computation with 4x4 register tiles
            // Thread computes 8 rows x 8 cols = 64 elements
            // As 4 blocks of 4x4
            
            for (int im = 0; im < 2; im++) {
                for (int jm = 0; jm < 2; jm++) {
                    // 4x4 block at (thread_m + im*4, thread_n + jm*4)
                    for (int i = 0; i < 4; i++) {
                        double a_val = A_row_shared[thread_m + im * 4 + i][kk];
                        for (int j = 0; j < 4; j++) {
                            accum[im * 4 + i][jm * 4 + j] += a_val * A_col_shared[thread_n + jm * 4 + j][kk];
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to C with lower triangle check
    for (int im = 0; im < 2; im++) {
        for (int jm = 0; jm < 2; jm++) {
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    int global_i = row_start + thread_m + im * 4 + i;
                    int global_j = col_start + thread_n + jm * 4 + j;
                    
                    // Lower triangle: only write if global_i >= global_j
                    if (global_i < N && global_j < N && global_i >= global_j) {
                        double val = alpha * accum[im * 4 + i][jm * 4 + j];
                        if (beta != 0.0) {
                            val += beta * C[global_i + global_j * ldc];
                        }
                        C[global_i + global_j * ldc] = val;
                    }
                }
            }
        }
    }
}

// Optimized version with better thread utilization
__global__ void dsyrk_kernel_v2(
    int N, int K,
    double alpha,
    const double *A, int lda,
    double beta,
    double *C, int ldc)
{
    const int BM = 64;  // tile size in M dimension (rows of C)
    const int BN = 64;  // tile size in N dimension (cols of C)  
    const int BK = 8;   // tile size in K dimension
    
    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;
    
    // Skip upper triangle
    if (block_row * BM < block_col * BN) {
        return;
    }
    
    const int row_start = block_row * BM;
    const int col_start = block_col * BN;
    
    // 16x16 thread block = 256 threads
    const int tx = threadIdx.x % 16;  // 0-15
    const int ty = threadIdx.x / 16;  // 0-15
    
    // Each thread computes a 4x4 output tile
    // Thread (tx, ty) computes rows [ty*4, ty*4+3] and cols [tx*4, tx*4+3]
    
    __shared__ double As[BM][BK];
    __shared__ double Bs[BN][BK];  // Actually A^T, so this is A[col_start:col_start+BN, :]
    
    double Creg[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            Creg[i][j] = 0.0;
        }
    }
    
    for (int k = 0; k < K; k += BK) {
        // Load A tile: BM x BK
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int load_row = ty * 4 + t;
            int load_col = tx;
            if (load_row < BM && load_col < BK) {
                int a_row = row_start + load_row;
                int a_k = k + load_col;
                if (a_row < N && a_k < K) {
                    As[load_row][load_col] = A[a_row + a_k * lda];
                } else {
                    As[load_row][load_col] = 0.0;
                }
            }
        }
        
        // Load B tile (A^T part): BN x BK
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int load_row = tx * 4 + t;  // transpose for coalescing
            int load_col = ty;
            if (load_row < BN && load_col < BK) {
                int a_col = col_start + load_row;
                int a_k = k + load_col;
                if (a_col < N && a_k < K) {
                    Bs[load_row][load_col] = A[a_col + a_k * lda];
                } else {
                    Bs[load_row][load_col] = 0.0;
                }
            }
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            double Ar[4];
            double Br[4];
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                Ar[i] = As[ty * 4 + i][kk];
                Br[i] = Bs[tx * 4 + i][kk];
            }
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    Creg[i][j] += Ar[i] * Br[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write to C with lower triangle check
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int global_i = row_start + ty * 4 + i;
            int global_j = col_start + tx * 4 + j;
            
            if (global_i < N && global_j < N && global_i >= global_j) {
                double val = beta * Creg[i][j];
                if (beta != 0.0) {
                    val += alpha * C[global_i + global_j * ldc];
                }
                C[global_i + global_j * ldc] = val;
            }
        }
    }
}

// Even more optimized version with better memory access patterns
__global__ void dsyrk_kernel_v3(
    int N, int K,
    double alpha,
    const double *A, int lda,
    double beta,
    double *C, int ldc)
{
    const int BM = 64;
    const int BN = 64; 
    const int BK = 8;
    
    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;
    
    // Skip upper triangle
    if (block_row * BM < block_col * BN) {
        return;
    }
    
    const int row_start = block_row * BM;
    const int col_start = block_col * BN;
    
    const int tx = threadIdx.x % 16;
    const int ty = threadIdx.x / 16;
    
    __shared__ double As[BM][BK + 1];  // +1 for bank conflict avoidance
    __shared__ double Bs[BN][BK + 1];
    
    double Creg[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            Creg[i][j] = 0.0;
        }
    }
    
    // Preload first tile
    for (int k = 0; k < K; k += BK) {
        // Load A: coalesced access - each thread loads 4 consecutive elements
        int a_row = row_start + threadIdx.x / 2;
        int a_k = k + (threadIdx.x % 2) * 4;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (a_row < N && a_k + i < K) {
                As[threadIdx.x / 2][(threadIdx.x % 2) * 4 + i] = A[a_row + (a_k + i) * lda];
            } else {
                As[threadIdx.x / 2][(threadIdx.x % 2) * 4 + i] = 0.0;
            }
        }
        
        // Load B (second A): 
        int b_row = col_start + threadIdx.x / 2;
        int b_k = k + (threadIdx.x % 2) * 4;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            if (b_row < N && b_k + i < K) {
                Bs[threadIdx.x / 2][(threadIdx.x % 2) * 4 + i] = A[b_row + (b_k + i) * lda];
            } else {
                Bs[threadIdx.x / 2][(threadIdx.x % 2) * 4 + i] = 0.0;
            }
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            double Ar[4], Br[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                Ar[i] = As[ty * 4 + i][kk];
                Br[i] = Bs[tx * 4 + i][kk];
            }
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    Creg[i][j] += Ar[i] * Br[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int global_i = row_start + ty * 4 + i;
            int global_j = col_start + tx * 4 + j;
            
            if (global_i < N && global_j < N && global_i >= global_j) {
                double val = beta * Creg[i][j];
                if (beta != 0.0) {
                    val += alpha * C[global_i + global_j * ldc];
                }
                C[global_i + global_j * ldc] = val;
            }
        }
    }
}

// Final optimized version
__global__ void dsyrk_kernel_final(
    int N, int K,
    double alpha,
    const double *A, int lda,
    double beta,
    double *C, int ldc)
{
    // Tile sizes
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    
    const int block_row = blockIdx.x;
    const int block_col = blockIdx.y;
    
    // Skip upper triangle blocks entirely
    if (block_row * BM < block_col * BN) {
        return;
    }
    
    const int row_start = block_row * BM;
    const int col_start = block_col * BN;
    
    // Thread mapping: 256 threads = 16x16
    const int tx = threadIdx.x & 15;      // % 16
    const int ty = threadIdx.x >> 4;      // / 16
    
    // Shared memory with padding to avoid bank conflicts
    __shared__ double As[BM][BK + 1];
    __shared__ double Bs[BN][BK + 1];
    
    // Register accumulator: 4x4 per thread
    double Creg[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            Creg[i][j] = 0.0;
        }
    }
    
    // Double buffering registers for next tile
    double Ar[4], Br[4];
    
    for (int k = 0; k < K; k += BK) {
        // Load A tile (BM x BK) - coalesced
        // Each thread loads 2 elements from 4 different rows
        const int load_row_A = threadIdx.x >> 2;           // 0-63
        const int load_col_A = (threadIdx.x & 3) << 1;     // 0, 2, 4, 6
        
        if (load_row_A < BM) {
            int a_row = row_start + load_row_A;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int a_k = k + load_col_A + i;
                if (a_row < N && a_k < K) {
                    As[load_row_A][load_col_A + i] = A[a_row + a_k * lda];
                } else {
                    As[load_row_A][load_col_A + i] = 0.0;
                }
            }
        }
        
        // Load B tile (BN x BK) - coalesced
        const int load_row_B = threadIdx.x >> 2;
        const int load_col_B = (threadIdx.x & 3) << 1;
        
        if (load_row_B < BN) {
            int b_row = col_start + load_row_B;
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                int b_k = k + load_col_B + i;
                if (b_row < N && b_k < K) {
                    Bs[load_row_B][load_col_B + i] = A[b_row + b_k * lda];
                } else {
                    Bs[load_row_B][load_col_B + i] = 0.0;
                }
            }
        }
        
        __syncthreads();
        
        // Compute: 4x4 register tiling
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load from shared to registers
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                Ar[i] = As[(ty << 2) + i][kk];
                Br[i] = Bs[(tx << 2) + i][kk];
            }
            
            // FMA
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    Creg[i][j] = fma(Ar[i], Br[j], Creg[i][j]);
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write to global memory with lower triangle check
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int global_i = row_start + (ty << 2) + i;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int global_j = col_start + (tx << 2) + j;
            
            if (global_i < N && global_j < N && global_i >= global_j) {
                double val = beta * Creg[i][j];
                if (beta != 0.0) {
                    val += alpha * C[global_i + global_j * ldc];
                }
                C[global_i + global_j * ldc] = val;
            }
        }
    }
}

cudaError_t Dsyrk(
    int N, int K,
    double alpha,
    double const *A, int lda,
    double beta,
    double *C, int ldc)
{
    if (N <= 0 || K <= 0) {
        return cudaSuccess;
    }
    
    const int BM = 64;
    const int BN = 64;
    
    // Grid size: cover lower triangle of N x N matrix
    int grid_m = (N + BM - 1) / BM;
    int grid_n = (N + BN - 1) / BN;
    
    dim3 block(256);
    dim3 grid(grid_m, grid_n);
    
    dsyrk_kernel_final<<<grid, block>>>(N, K, alpha, A, lda, beta, C, ldc);
    
    return cudaGetLastError();
}
