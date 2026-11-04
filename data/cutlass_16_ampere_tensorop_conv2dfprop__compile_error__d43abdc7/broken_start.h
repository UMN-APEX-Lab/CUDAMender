#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Conv2D Forward Propagation (Fprop) with half-precision input and single-precision output, NHWC layout.
// output = alpha * Conv2d(input, filter) + beta * output

// Tile dimensions for thread block
#define TILE_P 8
#define TILE_Q 8
#define TILE_K 64

// Number of threads per block
#define THREADS_PER_BLOCK 256

// Unroll factors
#define UNROLL_R 3
#define UNROLL_S 3

// Shared memory tile dimensions
#define SMEM_C 32  // Input channels per load
#define SMEM_H (TILE_P + UNROLL_R - 1)  // Height of input tile
#define SMEM_W (TILE_Q + UNROLL_S - 1)  // Width of input tile

__global__ void Conv2dFpropF16Kernel(
    __half const * __restrict__ input,
    __half const * __restrict__ filter,
    float * __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q,
    float alpha, float beta)
{
    // Block coordinates
    const int block_n = blockIdx.z;
    const int block_p = (blockIdx.y * TILE_P);
    const int block_q = (blockIdx.x * TILE_Q);
    const int block_k = (blockIdx.y * gridDim.x + blockIdx.x) * TILE_K / (gridDim.x * gridDim.y) * 0 + blockIdx.x * TILE_K;  // Simplified: each block handles TILE_K output channels
    
    // Actually, let's use a simpler mapping: blockIdx.x for Q, blockIdx.y for P, blockIdx.z for N
    // And handle K in tiles within the block or use blockIdx.w if available
    
    // Re-coordinate: use 3D grid: (Q_blocks, P_blocks, N)
    // Each block computes TILE_P x TILE_Q x TILE_K output elements
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // Each thread computes a subset of the output tile
    // We'll use 2D tiling in P-Q and handle K as the inner dimension
    
    // Position within tile
    const int tp = tid / (TILE_Q * 2);  // 4 threads per Q, 2 per K group
    const int tq = (tid % (TILE_Q * 2)) / 2;
    const int tk_group = tid % 2;  // Which group of 32 K channels
    
    // Actually, let's use a cleaner approach: each warp handles 32 K channels
    // 8 warps = 256 threads, handle TILE_K=64 K channels (2 groups of 32 per warp or 2 K per thread)
    
    const int k_per_thread = TILE_K / THREADS_PER_BLOCK;  // 64/256 = 0.25, not good
    
    // Better: each thread handles 1 output pixel and 4 K channels
    // 64 threads for P*Q = 64, each handles 4 K = 256 threads total
    
    const int pq_per_block = TILE_P * TILE_Q;  // 64
    const int k_per_thread = TILE_K / (THREADS_PER_BLOCK / pq_per_block);  // 64 / 4 = 16? No...
    
    // Let me recalculate: 256 threads, need to cover 8*8*64 = 4096 output elements
    // Each thread handles 4096/256 = 16 output elements
    // Layout: each thread handles 4x4 in K and 1x1 in P-Q, or 2x2 in P-Q and 4 in K, etc.
    
    // Simpler: 64 threads handle the 8x8 spatial tile (each handles 1 pixel)
    // 4 groups of 64 threads handle 4 slices of K (16 channels each)
    // Total: 256 threads, K per group = 64/4 = 16
    
    const int num_pq_threads = 64;  // 8x8 spatial
    const int num_k_groups = THREADS_PER_BLOCK / num_pq_threads;  // 4
    const int k_per_group = TILE_K / num_k_groups;  // 16
    
    const int pq_tid = tid % num_pq_threads;
    const int k_group = tid / num_pq_threads;
    
    const int local_p = pq_tid / TILE_Q;
    const int local_q = pq_tid % TILE_Q;
    
    const int global_p = block_p + local_p;
    const int global_q = block_q + local_q;
    const int global_n = block_n;
    
    // Starting K for this thread's group
    const int k_start = k_group * k_per_group;
    
    // Accumulators (float for precision)
    float acc[16];  // Max 16 K channels per thread
    #pragma unroll
    for (int i = 0; i < k_per_group; i++) {
        acc[i] = 0.0f;
    }
    
    // Shared memory for input tile
    // We need to load input for the receptive field
    __shared__ __half smem_input[SMEM_H][SMEM_W][SMEM_C];  // Too big? 10*10*32*2 = 6.4KB
    
    // Each thread loads part of the input
    const int smem_size = SMEM_H * SMEM_W * SMEM_C;
    const int loads_per_thread = (smem_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Global memory indices for input
    // Input base for this block
    const int in_h_base = block_p * stride_h - pad_h;
    const int in_w_base = block_q * stride_w - pad_w;
    
    // Main loop over C in tiles
    for (int c_tile = 0; c_tile < C; c_tile += SMEM_C) {
        // Load input tile to shared memory
        #pragma unroll
        for (int l = 0; l < loads_per_thread; l++) {
            int idx = tid + l * THREADS_PER_BLOCK;
            if (idx < smem_size) {
                int sc = idx % SMEM_C;
                int sw = (idx / SMEM_C) % SMEM_W;
                int sh = idx / (SMEM_C * SMEM_W);
                
                int h = in_h_base + sh;
                int w = in_w_base + sw;
                int c = c_tile + sc;
                
                __half val = __half(0.0f);
                if (h >= 0 && h < H && w >= 0 && w < W && c < C) {
                    val = input[((global_n * H + h) * W + w) * C + c];
                }
                smem_input[sh][sw][sc] = val;
            }
        }
        __syncthreads();
        
        // Compute convolution for this C tile
        // Each thread computes its assigned output pixels
        
        if (global_p < P && global_q < Q) {
            // Loop over R, S
            #pragma unroll
            for (int r = 0; r < R; r++) {
                #pragma unroll
                for (int s = 0; s < S; s++) {
                    int h = local_p * stride_h + r;
                    int w = local_q * stride_w + s;
                    
                    // Check bounds for shared memory access
                    if (h < SMEM_H && w < SMEM_W) {
                        // Load filter for this r,s and all K in our group
                        // Filter is [K][R][S][C]
                        
                        #pragma unroll
                        for (int kk = 0; kk < k_per_group; kk++) {
                            int k = k_start + kk;
                            if (k < K) {
                                float sum = 0.0f;
                                
                                // Inner loop over C in shared memory
                                #pragma unroll
                                for (int sc = 0; sc < SMEM_C && (c_tile + sc) < C; sc++) {
                                    __half in_val = smem_input[h][w][sc];
                                    
                                    // Filter access: filter[((k * R + r) * S + s) * C + (c_tile + sc)]
                                    __half f_val = filter[((k * R + r) * S + s) * C + (c_tile + sc)];
                                    
                                    sum += __half2float(in_val) * __half2float(f_val);
                                }
                                acc[kk] += sum;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write output with alpha/beta scaling
    if (global_p < P && global_q < Q) {
        #pragma unroll
        for (int kk = 0; kk < k_per_group; kk++) {
            int k = k_start + kk;
            if (k < K) {
                float out_val = alpha * acc[kk];
                
                if (beta != 0.0f) {
                    float old_val = output[((global_n * P + global_p) * Q + global_q) * K + k];
                    out_val += beta * old_val;
                }
                
                output[((global_n * P + global_p) * Q + global_q) * K + k] = out_val;
            }
        }
    }
}

// Optimized version using more efficient memory access patterns
__global__ void Conv2dFpropF16KernelOpt(
    __half const * __restrict__ input,
    __half const * __restrict__ filter,
    float * __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q,
    float alpha, float beta)
{
    // Block tiling: each block computes a tile of output
    // Tile size: 8x8 spatial, 64 channels
    
    const int block_n = blockIdx.z;
    const int block_p = (blockIdx.y / ((Q + 7) / 8)) * 8;  // Incorrect, fix below
    
    // Simpler 3D grid: blockIdx.x for Q tile, blockIdx.y for P tile, blockIdx.z for N
    const int tile_q = 8;
    const int tile_p = 8;
    const int tile_k = 64;
    
    const int num_q_tiles = (Q + tile_q - 1) / tile_q;
    
    const int block_q_tile = blockIdx.x % num_q_tiles;
    const int block_p_tile = blockIdx.x / num_q_tiles;
    const int block_k_tile = blockIdx.y;  // Additional dimension for K
    
    const int base_q = block_q_tile * tile_q;
    const int base_p = block_p_tile * tile_p;
    const int base_k = block_k_tile * tile_k;
    
    const int tid = threadIdx.x;
    
    // 256 threads: organize as 8x8 for spatial, with 4 for K
    // Actually: 64 threads for 8x8 spatial (1 per pixel), 4-way K unroll
    
    const int local_pq = tid % 64;  // 0-63 for spatial
    const int local_k = tid / 64;   // 0-3 for K offset (16 channels each, total 64)
    
    const int local_p = local_pq / 8;
    const int local_q = local_pq % 8;
    
    const int global_p = base_p + local_p;
    const int global_q = base_q + local_q;
    const int global_n = block_n;
    
    const int k_start = base_k + local_k * 16;
    const int k_end = min(k_start + 16, K);
    
    // Accumulators
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    
    // Check if this thread has valid work
    bool valid = (global_p < P) && (global_q < Q) && (k_start < K);
    
    // Input position
    const int in_h_base = base_p * stride_h - pad_h;
    const int in_w_base = base_q * stride_w - pad_w;
    
    // Shared memory for input: need (8*stride+2) x (8*stride+2) typically
    // For stride=1, pad=1, R=S=3: need 10x10
    const int smem_h = tile_p * stride_h + R - 1;
    const int smem_w = tile_q * stride_w + S - 1;
    
    // Use dynamic shared memory or limit size
    __shared__ __half smem[10 * 10 * 32];  // Fixed size for common case
    
    // Process C in tiles of 32
    for (int c_base = 0; c_base < C; c_base += 32) {
        // Load input to shared memory
        // Each thread loads multiple elements
        const int smem_total = smem_h * smem_w * 32;
        const int loads = (smem_total + 255) / 256;
        
        #pragma unroll
        for (int l = 0; l < loads; l++) {
            int idx = tid + l * 256;
            if (idx < smem_total) {
                int sc = idx % 32;
                int tmp = idx / 32;
                int sw = tmp % smem_w;
                int sh = tmp / smem_w;
                
                int h = in_h_base + sh;
                int w = in_w_base + sw;
                int c = c_base + sc;
                
                __half val = __half(0.0f);
                if (h >= 0 && h < H && w >= 0 && w < W && c < C) {
                    val = input[((global_n * H + h) * W + w) * C + c];
                }
                smem[idx] = val;
            }
        }
        __syncthreads();
        
        if (valid) {
            // Compute for each R,S
            #pragma unroll
            for (int r = 0; r < R; r++) {
                #pragma unroll
                for (int s = 0; s < S; s++) {
                    int ih = local_p * stride_h + r;
                    int iw = local_q * stride_w + s;
                    
                    // Access input from shared memory
                    #pragma unroll
                    for (int sc = 0; sc < 32 && (c_base + sc) < C; sc++) {
                        int smem_idx = ((ih * smem_w) + iw) * 32 + sc;
                        __half in_val = smem[smem_idx];
                        float in_f = __half2float(in_val);
                        
                        // Load filter and multiply-add for each K
                        int c = c_base + sc;
                        
                        #pragma unroll
                        for (int kk = 0; kk < 16 && (k_start + kk) < K; kk++) {
                            int k = k_start + kk;
                            __half f_val = filter[((k * R + r) * S + s) * C + c];
                            acc[kk] += in_f * __half2float(f_val);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    if (valid) {
        #pragma unroll
        for (int kk = 0; kk < 16 && (k_start + kk) < K; kk++) {
            int k = k_start + kk;
            int out_idx = ((global_n * P + global_p) * Q + global_q) * K + k;
            
            float result = alpha * acc[kk];
            if (beta != 0.0f) {
                result += beta * output[out_idx];
            }
            output[out_idx] = result;
        }
    }
}

// Most optimized version with proper grid setup
__global__ void Conv2dFpropF16KernelV2(
    __half const * __restrict__ input,
    __half const * __restrict__ filter,
    float * __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q,
    float alpha, float beta)
{
    // Grid: (Q_tiles, P_tiles * K_tiles, N)
    // Actually: use 2D grid with K_tiles in y, (P_tiles * Q_tiles) computed from x
    
    const int tile_p = 4;
    const int tile_q = 8;
    const int tile_k = 128;
    
    const int num_q_tiles = (Q + tile_q - 1) / tile_q;
    const int num_p_tiles = (P + tile_p - 1) / tile_p;
    
    // blockIdx.x: linear index of PQ tile
    // blockIdx.y: K tile
    // blockIdx.z: N
    
    const int pq_tile_idx = blockIdx.x;
    const int k_tile_idx = blockIdx.y;
    const int n = blockIdx.z;
    
    const int p_tile = pq_tile_idx / num_q_tiles;
    const int q_tile = pq_tile_idx % num_q_tiles;
    
    const int base_p = p_tile * tile_p;
    const int base_q = q_tile * tile_q;
    const int base_k = k_tile_idx * tile_k;
    
    const int tid = threadIdx.x;  // 0-255
    
    // Thread organization: 4 warps
    // Each warp: 32 threads
    // Warp 0-1: handle K channels 0-63
    // Warp 2-3: handle K channels 64-127
    
    const int warp_id = tid >> 5;      // 0-7
    const int lane_id = tid & 31;      // 0-31
    
    // Each warp handles 16 K channels (4 warps per 64 K group)
    // Actually with 8 warps and 128 K: each warp handles 16 K
    
    const int k_per_warp = tile_k / 8;  // 16
    const int k_warp_start = base_k + warp_id * k_per_warp;
    
    // Within warp: 32 threads handle spatial tile
    // 4x8 = 32, so each thread handles 1 spatial pixel
    
    const int local_p = lane_id >> 3;   // 0-3 (4 rows)
    const int local_q = lane_id & 7;    // 0-7 (8 cols)
    
    const int global_p = base_p + local_p;
    const int global_q = base_q + local_q;
    
    // Check validity
    bool valid = (global_p < P) && (global_q < Q) && (k_warp_start < K);
    
    // Accumulators: 16 floats per thread
    float acc[16];
    #pragma unroll
    for (int i = 0; i < k_per_warp; i++) acc[i] = 0.0f;
    
    // Input loading setup
    const int in_h_base = base_p * stride_h - pad_h;
    const int in_w_base = base_q * stride_w - pad_w;
    
    // Shared memory: input tile + filter tile
    // Input: need (4*stride+R-1) x (8*stride+S-1) x C_tile
    // Typical: stride=1, R=S=3, pad=1: need 6x10
    
    const int smem_h = tile_p * stride_h + R - 1;
    const int smem_w = tile_q * stride_w + S - 1;
    const int c_tile = 32;
    
    __shared__ __half smem_input[6 * 10 * 32];  // ~3.8KB
    
    // Process C in tiles
    for (int c_base = 0; c_base < C; c_base += c_tile) {
        // Load input to shared memory
        // 256 threads, smem size = smem_h * smem_w * c_tile
        const int smem_size = smem_h * smem_w * c_tile;
        
        #pragma unroll
        for (int idx = tid; idx < smem_size; idx += 256) {
            int sc = idx % c_tile;
            int tmp = idx / c_tile;
            int sw = tmp % smem_w;
            int sh = tmp / smem_w;
            
            int h = in_h_base + sh;
            int w = in_w_base + sw;
            int c = c_base + sc;
            
            __half val = __half(0.0f);
            if (h >= 0 && h < H && w >= 0 && w < W && c < C) {
                val = input[((n * H + h) * W + w) * C + c];
            }
            smem_input[idx] = val;
        }
        __syncthreads();
        
        if (valid) {
            // Compute convolution
            #pragma unroll
            for (int r = 0; r < R; r++) {
                #pragma unroll
                for (int s = 0; s < S; s++) {
                    int ih = local_p * stride_h + r;
                    int iw = local_q * stride_w + s;
                    
                    // Bounds check for smem access
                    if (ih < smem_h && iw < smem_w) {
                        #pragma unroll
                        for (int sc = 0; sc < c_tile && (c_base + sc) < C; sc++) {
                            int smem_idx = ((ih * smem_w) + iw) * c_tile + sc;
                            float in_val = __half2float(smem_input[smem_idx]);
                            
                            int c = c_base + sc;
                            
                            // Load filter and accumulate
                            #pragma unroll
                            for (int kk = 0; kk < k_per_warp; kk++) {
                                int k = k_warp_start + kk;
                                if (k < K) {
                                    __half f_val = filter[((k * R + r) * S + s) * C + c];
                                    acc[kk] += in_val * __half2float(f_val);
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    if (valid) {
        #pragma unroll
        for (int kk = 0; kk < k_per_warp; kk++) {
            int k = k_warp_start + kk;
            if (k < K) {
                int out_idx = ((n * P + global_p) * Q + global_q) * K + k;
                float res = alpha * acc[kk];
                if (beta != 0.0f) {
                    res += beta * output[out_idx];
                }
                output[out_idx] = res;
            }
        }
    }
}

// Simpler, more robust version
__global__ void Conv2dFpropF16KernelSimple(
    __half const * __restrict__ input,
    __half const * __restrict__ filter,
    float * __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q,
    float alpha, float beta)
{
    // Each thread computes one output element (n,p,q,k)
    // Grid is 1D or 2D, we map to output space
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_outputs = N * P * Q * K;
    
    if (tid >= total_outputs) return;
    
    // Decode tid to n,p,q,k
    int tmp = tid;
    int k = tmp % K; tmp /= K;
    int q = tmp % Q; tmp /= Q;
    int p = tmp % P; tmp /= P;
    int n = tmp;
    
    // Compute convolution
    float sum = 0.0f;
    
    for (int r = 0; r < R; r++) {
        for (int s = 0; s < S; s++) {
            int h = p * stride_h + r - pad_h;
            int w = q * stride_w + s - pad_w;
            
            if (h >= 0 && h < H && w >= 0 && w < W) {
                for (int c = 0; c < C; c++) {
                    __half in_val = input[((n * H + h) * W + w) * C + c];
                    __half f_val = filter[((k * R + r) * S + s) * C + c];
                    sum += __half2float(in_val) * __half2float(f_val);
                }
            }
        }
    }
    
    int out_idx = ((n * P + p) * Q + q) * K + k;
    float result = alpha * sum;
    if (beta != 0.0f) {
        result += beta * output[out_idx];
    }
    output[out_idx] = result;
}

// Optimized tiled version
__global__ void Conv2dFpropF16KernelTiled(
    __half const * __restrict__ input,
    __half const * __restrict__ filter,
    float * __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q,
    float alpha, float beta)
{
    // Tile configuration
    const int TP = 8;   // Tile in P
    const int TQ = 8;   // Tile in Q  
    const int TK = 32;  // Tile in K (per thread)
    const int TC = 16;  // Tile in C (shared memory)
    
    // Block computes TP x TQ x TK output elements
    // Using 256 threads: 8 warps
    
    const int block_n = blockIdx.z;
    const int block_p = blockIdx.y * TP;
    const int block_q = (blockIdx.x * TQ);
    const int block_k = (blockIdx.x / ((Q + TQ - 1) / TQ)) * TK;  // This needs fixing
    
    // Better grid: 3D with explicit dimensions
    // Actually use: blockIdx.x for Q, blockIdx.y for P, blockIdx.z for N
    // And handle K by having each block compute all K (too slow) or use threadIdx
    
    // Let me use: each warp handles 32 K channels, 8 warps handle 256 K
    // But K might be smaller, so adjust
    
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Position in tile
    const int num_pq_per_warp = (TP * TQ + 7) / 8;  // Distribute spatial across warps
    
    // Actually simpler: 64 threads for 8x8 spatial, rest for K
    const int pq_thread = threadIdx.x % 64;
    const int k_group = threadIdx.x / 64;  // 0-3, each handles TK/4 = 8 K channels... too small
    
    // Try: 32 threads for spatial (4x8), 8 groups for K (4 channels each)
    const int sp_thread = threadIdx.x % 32;
    const int k_group_id = threadIdx.x / 32;  // 0-7
    
    const int local_p = sp_thread / 8;
    const int local_q = sp_thread % 8;
    
    const int k_per_group = TK / 8;  // 4
    const int k_start = k_group_id * k_per_group;
    
    const int global_p = block_p + local_p;
    const int global_q = block_q + local_q;
    const int global_n = block_n;
    
    // Accumulators
    float acc[4];
    #pragma unroll
    for (int i = 0; i < k_per_group; i++) acc[i] = 0.0f;
    
    // Shared memory for input
    const int smem_h = TP * stride_h + R - 1;
    const int smem_w = TQ * stride_w + S - 1;
    
    __shared__ __half smem[16 * 16 * 16];  // Conservative size
    
    const int in_h_base = block_p * stride_h - pad_h;
    const int in_w_base = block_q * stride_w - pad_w;
    
    for (int c_base = 0; c_base < C; c_base += TC) {
        // Load input
        #pragma unroll
        for (int idx = threadIdx.x; idx < smem_h * smem_w * TC; idx += 256) {
            int sc = idx % TC;
            int tmp = idx / TC;
            int sw = tmp % smem_w;
            int sh = tmp / smem_w;
            
            int h = in_h_base + sh;
            int w = in_w_base + sw;
            int c = c_base + sc;
            
            __half val = __half(0.0f);
            if (h >= 0 && h < H && w >= 0 && w < W && c < C) {
                val = input[((global_n * H + h) * W + w) * C + c];
            }
            // Simple linear indexing
            smem[idx] = val;
        }
        __syncthreads();
        
        // Compute
        if (global_p < P && global_q < Q) {
            #pragma unroll
            for (int r = 0; r < R; r++) {
                #pragma unroll
                for (int s = 0; s < S; s++) {
                    int ih = local_p * stride_h + r;
                    int iw = local_q * stride_w + s;
                    
                    if (ih < smem_h && iw < smem_w) {
                        #pragma unroll
                        for (int sc = 0; sc < TC && c_base + sc < C; sc++) {
                            int smem_idx = ((ih * smem_w) + iw) * TC + sc;
                            float in_val = __half2float(smem[smem_idx]);
                            int c = c_base + sc;
                            
                            #pragma unroll
                            for (int kk = 0; kk < k_per_group; kk++) {
                                int k = k_start + kk;
                                if (k < K) {
                                    __half f_val = filter[((k * R + r) * S + s) * C + c];
                                    acc[kk] += in_val * __half2float(f_val);
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write output
    if (global_p < P && global_q < Q) {
        #pragma unroll
        for (int kk = 0; kk < k_per_group; kk++) {
            int k = k_start + kk;
            if (k < K) {
                int idx = ((global_n * P + global_p) * Q + global_q) * K + k;
                float res = alpha * acc[kk];
                if (beta != 0.0f) res += beta * output[idx];
                output[idx] = res;
            }
        }
    }
}

// Final optimized kernel with correct grid mapping
__global__ void __launch_bounds__(256, 2) Conv2dFpropF16KernelFinal(
    __half const * __restrict__ input,
    __half const * __restrict__ filter,
    float * __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q,
    float alpha, float beta)
{
    // Configuration: each block computes 8x8 spatial tile with 64 K channels
    // Grid: (n, p_tile, q_tile) with k handled in blocks or threads
    
    const int TP = 8;
    const int TQ = 8;
    const int TK = 64;
    
    // 3D grid: (Q_tiles, P_tiles, N) - each block handles full K or K-tile
    // Let's have each block compute TK=64 K channels, need K/64 blocks in some dimension
    
    // Alternative: 4D grid simulation
    // blockIdx.x: Q tile
    // blockIdx.y: P tile * K_tiles + K_tile
    // blockIdx.z: N
    
    const int num_k_tiles = (K + TK - 1) / TK;
    const int num_q_tiles = (Q + TQ - 1) / TQ;
    
    const int q_tile = blockIdx.x % num_q_tiles;
    const int p_tile = blockIdx.x / num_q_tiles;
    const int k_tile = blockIdx.y;
    const int n = blockIdx.z;
    
    const int base_q = q_tile * TQ;
    const int base_p = p_tile * TP;
    const int base_k = k_tile * TK;
    
    // Thread organization
    const int tid = threadIdx.x;
    
    // 256 threads: 4x64 organization
    // - 64 threads for 8x8 spatial (each thread handles 1 pixel)
    // - 4 groups for K (each group handles 16 K channels: 64/4=16)
    
    const int spatial_tid = tid % 64;      // Which spatial pixel: 0-63
    const int k_group = tid / 64;          // Which K group: 0-3
    
    const int local_p = spatial_tid / TQ;  // 0-7
    const int local_q = spatial_tid % TQ;  // 0-7
    
    const int k_per_thread = TK / 4;       // 16
    const int k_start = base_k + k_group * k_per_thread;
    
    const int global_p = base_p + local_p;
    const int global_q = base_q + local_q;
    
    // Early exit if out of bounds
    if (global_p >= P || global_q >= Q || k_start >= K) return;
    
    // Accumulators
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    
    // Input tile dimensions in shared memory
    const int smem_h = TP * stride_h + R - 1;
    const int smem_w = TQ * stride_w + S - 1;
    const int TC = 32;  // C tile size
    
    // Use extern shared memory for flexibility
    extern __shared__ __half smem[];
    
    const int in_h_base = base_p * stride_h - pad_h;
    const int in_w_base = base_q * stride_w - pad_w;
    
    // Main loop over C
    for (int c_base = 0; c_base < C; c_base += TC) {
        // Load input tile to shared memory
        const int smem_size = smem_h * smem_w * TC;
        
        #pragma unroll
        for (int idx = tid; idx < smem_size; idx += 256) {
            int sc = idx % TC;
            int tmp = idx / TC;
            int sw = tmp % smem_w;
            int sh = tmp / smem_w;
            
            int h = in_h_base + sh;
            int w = in_w_base + sw;
            int c = c_base + sc;
            
            __half val = __half(0.0f);
            if (h >= 0 && h < H && w >= 0 && w < W && c < C) {
                val = input[((n * H + h) * W + w) * C + c];
            }
            smem[idx] = val;
        }
        __syncthreads();
        
        // Compute convolution
        #pragma unroll
        for (int r = 0; r < R; r++) {
            #pragma unroll
            for (int s = 0; s < S; s++) {
                int ih = local_p * stride_h + r;
                int iw = local_q * stride_w + s;
                
                // Check bounds
                if (ih >= 0 && ih < smem_h && iw >= 0 && iw < smem_w) {
                    #pragma unroll
                    for (int sc = 0; sc < TC && c_base + sc < C; sc++) {
                        int smem_idx = ((ih * smem_w) + iw) * TC + sc;
                        float in_val = __half2float(smem[smem_idx]);
                        int c = c_base + sc;
                        
                        // Compute partial dot product for all K in this thread's group
                        #pragma unroll
                        for (int kk = 0; kk < k_per_thread; kk++) {
                            int k = k_start + kk;
                            if (k < K) {
                                // Filter: [K][R][S][C]
                                __half f_val = filter[((k * R + r) * S + s) * C + c];
                                acc[kk] += in_val * __half2float(f_val);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Write output with alpha/beta scaling
    #pragma unroll
    for (int kk = 0; kk < k_per_thread; kk++) {
        int k = k_start + kk;
        if (k < K) {
            int out_idx = ((n * P + global_p) * Q + global_q) * K + k;
            float result = alpha * acc[kk];
            if (beta != 0.0f) {
                result += beta * output[out_idx];
            }
            output[out_idx] = result;
        }
    }
}

cudaError_t Conv2dFpropF16(
    __half const *input, __half const *filter, float *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    float alpha, float beta)
{
    // Calculate output dimensions
    int P = (H + 2 * pad_h - R) / stride_h + 1;
    int Q = (W + 2 * pad_w - S) / stride_w + 1;
    
    // Choose kernel based on problem size
    // For small problems, use simple approach
    // For larger problems, use tiled approach
    
    const int total_output_size = N * P * Q * K;
    
    if (total_output_size <= 65536) {
        // Simple kernel for small problems
        const int threads = 256;
        const int blocks = (total_output_size + threads - 1) / threads;
        
        Conv2dFpropF16KernelSimple<<<blocks, threads>>>(
            input, filter, output,
            N, C, H, W, K, R, S,
            pad_h, pad_w, stride_h, stride_w,
            P, Q,
            alpha, beta);
    } else {
        // Tiled kernel for larger problems
        const int TP = 8;
        const int TQ = 8;
        const int TK = 64;
        
        const int num_q_tiles = (Q + TQ - 1) / TQ;
        const int num_p_tiles = (P + TP - 1) / TP;
        const int num_k_tiles = (K + TK - 1) / TK;
        
        // Grid: PQ tiles in x, K tiles in y, N in z
        // Actually: pack P and Q into x, K into y
        dim3 grid(num_q_tiles * num_p_tiles, num_k_tiles, N);
        dim3 block(256);
        
        // Calculate shared memory size
        const int smem_h = TP * stride_h + R - 1;
        const int smem_w = TQ * stride_w + S - 1;
        const int TC = 32;
        size_t smem_size = smem_h * smem_w * TC * sizeof(__half);
        
        // Clamp shared memory to reasonable size
        if (smem_size > 48 * 1024) {
            // Fall back to smaller tile
            // Or use simple kernel
            const int threads = 256;
            const int blocks = (total_output_size + threads - 1) / threads;
            
            Conv2dFpropF16KernelSimple<<<blocks, threads>>>(
                input, filter, output,
                N, C, H, W, K, R, S,
                pad_h, pad_w, stride_h, stride_w,
                P, Q,
                alpha, beta);
        } else {
            Conv2dFpropF16KernelFinal<<<grid, block, smem_size>>>(
                input, filter, output,
                N, C, H, W, K, R, S,
                pad_h, pad_w, stride_h, stride_w,
                P, Q,
                alpha, beta);
        }
    }
    
    return cudaGetLastError();
}
