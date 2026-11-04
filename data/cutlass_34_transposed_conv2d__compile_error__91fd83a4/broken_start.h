#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Transposed Conv2D (Deconvolution) - NHWC layout, FP16 I/O, FP32 accumulation
// output[n][h_out][w_out][k] = sum_{c,r,s} input[n][h][w][c] * filter[c][r][s][k]
// where h_out = h * stride_h + r - pad_h, w_out = w * stride_w + s - pad_w

#define TILE_H 8
#define TILE_W 8
#define TILE_K 64

__global__ void transposed_conv2d_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ filter,
    __half* __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int H_out, int W_out)
{
    // Block dimensions: (TILE_W, TILE_H, TILE_K/32) threads
    // Each block computes a TILE_H x TILE_W x TILE_K tile of output
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int tid_z = threadIdx.z;
    
    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int block_z = blockIdx.z;
    
    // Output tile coordinates
    const int w_out_base = block_x * TILE_W;
    const int h_out_base = block_y * TILE_H;
    const int k_base = block_z * TILE_K;
    
    const int n = blockIdx.z / ((H_out + TILE_H - 1) / TILE_H * ((W_out + TILE_W - 1) / TILE_W));
    
    // Thread computes partial sum for specific (h_out, w_out, k) within tile
    // Distribute work: each thread handles multiple K values
    const int k_per_thread = TILE_K / blockDim.z;
    const int k_start = k_base + tid_z * k_per_thread;
    
    // Shared memory for input and filter tiles
    __shared__ float s_input[TILE_H][TILE_W][4]; // Small cache for input values
    __shared__ float s_filter[4][4][TILE_K]; // Small filter cache
    
    // Accumulators in registers
    float accum[TILE_H][TILE_W][4]; // [h][w][k_local]
    #pragma unroll
    for (int h = 0; h < TILE_H; h++) {
        #pragma unroll
        for (int w = 0; w < TILE_W; w++) {
            #pragma unroll
            for (int kk = 0; kk < 4; kk++) {
                accum[h][w][kk] = 0.0f;
            }
        }
    }
    
    // Iterate over C, R, S dimensions
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < R; r++) {
            for (int s = 0; s < S; s++) {
                // Precompute which input (h,w) maps to this output region
                // For each output (h_out, w_out), find corresponding input (h,w)
                // h_out = h * stride_h + r - pad_h  =>  h = (h_out - r + pad_h) / stride_h
                
                // Load filter values for this (c,r,s) into shared memory
                // filter layout: [c][r][s][k] = ((c*R + r)*S + s)*K + k
                const int filter_base = ((c * R + r) * S + s) * K + k_base;
                
                // Each thread loads some filter values
                #pragma unroll
                for (int kk = 0; kk < k_per_thread; kk += 4) {
                    int k_idx = k_start + kk + (tid_x % 4);
                    if (k_idx < K && k_idx < k_base + TILE_K) {
                        s_filter[r % 4][s % 4][k_idx - k_base] = 
                            __half2float(filter[filter_base + (k_idx - k_base)]);
                    }
                }
                
                __syncthreads();
                
                // For each position in output tile, find contributing input
                #pragma unroll
                for (int h_tile = 0; h_tile < TILE_H; h_tile++) {
                    int h_out = h_out_base + h_tile;
                    if (h_out >= H_out) continue;
                    
                    // h = (h_out - r + pad_h) / stride_h, must be exact division
                    int h_num = h_out - r + pad_h;
                    if (h_num < 0 || h_num % stride_h != 0) continue;
                    int h_in = h_num / stride_h;
                    if (h_in < 0 || h_in >= H) continue;
                    
                    #pragma unroll
                    for (int w_tile = 0; w_tile < TILE_W; w_tile++) {
                        int w_out = w_out_base + w_tile;
                        if (w_out >= W_out) continue;
                        
                        int w_num = w_out - s + pad_w;
                        if (w_num < 0 || w_num % stride_w != 0) continue;
                        int w_in = w_num / stride_w;
                        if (w_in < 0 || w_in >= W) continue;
                        
                        // Load input value
                        // input: [n][h][w][c] = ((n*H + h)*W + w)*C + c
                        float in_val = __half2float(input[((n * H + h_in) * W + w_in) * C + c]);
                        
                        // Accumulate with filter values
                        #pragma unroll
                        for (int kk = 0; kk < k_per_thread; kk++) {
                            int k_idx = k_start + kk;
                            if (k_idx >= K) continue;
                            
                            float f_val = s_filter[r % 4][s % 4][k_idx - k_base];
                            accum[h_tile][w_tile][kk] += in_val * f_val;
                        }
                    }
                }
                
                __syncthreads();
            }
        }
    }
    
    // Write output
    #pragma unroll
    for (int h_tile = 0; h_tile < TILE_H; h_tile++) {
        int h_out = h_out_base + h_tile;
        if (h_out >= H_out) continue;
        
        #pragma unroll
        for (int w_tile = 0; w_tile < TILE_W; w_tile++) {
            int w_out = w_out_base + w_tile;
            if (w_out >= W_out) continue;
            
            #pragma unroll
            for (int kk = 0; kk < k_per_thread; kk++) {
                int k_idx = k_start + kk;
                if (k_idx >= K) continue;
                
                // output: [n][h_out][w_out][k] = ((n*H_out + h_out)*W_out + w_out)*K + k
                int out_idx = ((n * H_out + h_out) * W_out + w_out) * K + k_idx;
                output[out_idx] = __float2half(accum[h_tile][w_tile][kk]);
            }
        }
    }
}

// Optimized kernel using a scatter approach (input -> output)
__global__ void transposed_conv2d_scatter_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ filter,
    __half* __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int H_out, int W_out)
{
    // Each thread block processes a region of input and scatters to output
    // Grid: (N, H/TH, W/TW) blocks, each handling C channels
    
    const int n = blockIdx.x;
    const int h_block = blockIdx.y;
    const int w_block = blockIdx.z;
    
    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int num_threads = blockDim.x * blockDim.y * blockDim.z;
    
    // Tile sizes
    const int TH = 4;  // input tile height
    const int TW = 4;  // input tile width
    const int TC = 32; // channels per iteration
    
    const int h_start = h_block * TH;
    const int w_start = w_block * TW;
    
    // Shared memory for input tile and filter strip
    __shared__ float s_in[TH][TW][TC];
    __shared__ float s_filt[TC][R][S][4]; // filter for TC channels, all R,S, and 4 K values
    
    // Process C in chunks of TC
    for (int c_base = 0; c_base < C; c_base += TC) {
        // Load input tile into shared memory
        // Each thread loads multiple elements
        for (int idx = tid; idx < TH * TW * TC; idx += num_threads) {
            int cc = idx % TC;
            int ww = (idx / TC) % TW;
            int hh = idx / (TC * TW);
            
            int h_in = h_start + hh;
            int w_in = w_start + ww;
            int c_in = c_base + cc;
            
            float val = 0.0f;
            if (h_in < H && w_in < W && c_in < C) {
                val = __half2float(input[((n * H + h_in) * W + w_in) * C + c_in]);
            }
            s_in[hh][ww][cc] = val;
        }
        
        // Load filter for this channel chunk (each thread loads some K)
        for (int idx = tid; idx < TC * R * S * 4; idx += num_threads) {
            int k_off = idx % 4;
            int s = (idx / 4) % S;
            int r = (idx / (4 * S)) % R;
            int cc = idx / (4 * S * R);
            
            int c_filt = c_base + cc;
            int k_filt = (blockIdx.x * 4 + k_off) % K; // This needs proper indexing
            
            if (c_filt < C && k_filt < K) {
                // filter: [c][r][s][k]
                s_filt[cc][r][s][k_off] = __half2float(filter[((c_filt * R + r) * S + s) * K + k_filt]);
            }
        }
        
        __syncthreads();
        
        // Scatter: for each input pixel, add to all affected output pixels
        for (int hh = 0; hh < TH; hh++) {
            int h_in = h_start + hh;
            if (h_in >= H) continue;
            
            for (int ww = 0; ww < TW; ww++) {
                int w_in = w_start + ww;
                if (w_in >= W) continue;
                
                for (int r = 0; r < R; r++) {
                    int h_out = h_in * stride_h + r - pad_h;
                    if (h_out < 0 || h_out >= H_out) continue;
                    
                    for (int s = 0; s < S; s++) {
                        int w_out = w_in * stride_w + s - pad_w;
                        if (w_out < 0 || w_out >= W_out) continue;
                        
                        // Compute output index
                        int out_base = ((n * H_out + h_out) * W_out + w_out) * K;
                        
                        // Accumulate over channels
                        for (int cc = 0; cc < TC && c_base + cc < C; cc++) {
                            float in_val = s_in[hh][ww][cc];
                            
                            // Atomic add for each K (simplified - need proper K handling)
                            for (int k = tid; k < K; k += num_threads) {
                                float f_val = __half2float(filter[(((c_base + cc) * R + r) * S + s) * K + k]);
                                float prod = in_val * f_val;
                                
                                // Use atomicAdd for FP16 (requires casting)
                                unsigned int* ptr = (unsigned int*)&output[out_base + k];
                                // This is tricky with FP16, use FP32 atomic via bit manipulation or separate approach
                                
                                // Actually, let's use a reduction approach instead
                                // For now, direct write with race condition awareness
                                // Better: use shared memory reduction then atomic
                            }
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }
}

// Simpler direct compute kernel - each thread computes one output pixel for all K
__global__ void transposed_conv2d_direct_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ filter,
    __half* __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int H_out, int W_out)
{
    // Grid: (W_out, H_out, N) - each block handles all K for one output pixel
    // Or better: 2D grid for spatial, threads for K
    
    const int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    const int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;
    
    if (w_out >= W_out || h_out >= H_out || n >= N) return;
    
    // Each thread handles a subset of K channels
    const int k_per_thread = (K + blockDim.z - 1) / blockDim.z;
    const int k_start = threadIdx.z * k_per_thread;
    
    // Accumulators
    float acc[8]; // Assume k_per_thread <= 8
    for (int i = 0; i < k_per_thread && k_start + i < K; i++) {
        acc[i] = 0.0f;
    }
    
    // Iterate over all contributing input positions
    for (int r = 0; r < R; r++) {
        int h_num = h_out - r + pad_h;
        if (h_num < 0 || h_num % stride_h != 0) continue;
        int h_in = h_num / stride_h;
        if (h_in < 0 || h_in >= H) continue;
        
        for (int s = 0; s < S; s++) {
            int w_num = w_out - s + pad_w;
            if (w_num < 0 || w_num % stride_w != 0) continue;
            int w_in = w_num / stride_w;
            if (w_in < 0 || w_in >= W) continue;
            
            // Load input
            float in_val = __half2float(input[((n * H + h_in) * W + w_in) * C + 0]);
            
            // For each channel
            for (int c = 0; c < C; c++) {
                in_val = __half2float(input[((n * H + h_in) * W + w_in) * C + c]);
                
                // Load filter and accumulate
                // filter: [c][r][s][k]
                int f_base = ((c * R + r) * S + s) * K;
                
                for (int i = 0; i < k_per_thread && k_start + i < K; i++) {
                    float f_val = __half2float(filter[f_base + k_start + i]);
                    acc[i] += in_val * f_val;
                }
            }
        }
    }
    
    // Write output
    int out_base = ((n * H_out + h_out) * W_out + w_out) * K;
    for (int i = 0; i < k_per_thread && k_start + i < K; i++) {
        output[out_base + k_start + i] = __float2half(acc[i]);
    }
}

// Optimized version with better memory coalescing
__global__ void transposed_conv2d_optimized_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ filter,
    __half* __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int H_out, int W_out)
{
    // Block: (8, 8, 8) = 512 threads
    // Grid: ((W_out+7)/8, (H_out+7)/8, N*K/8)
    
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int tid_z = threadIdx.z;
    
    const int w_out = blockIdx.x * 8 + tid_x;
    const int h_out = blockIdx.y * 8 + tid_y;
    const int k_blk = blockIdx.z % ((K + 7) / 8);
    const int n = blockIdx.z / ((K + 7) / 8);
    
    const int k_out = k_blk * 8 + tid_z;
    
    if (w_out >= W_out || h_out >= H_out || n >= N || k_out >= K) return;
    
    float acc = 0.0f;
    
    // Loop over R, S, C
    for (int r = 0; r < R; r++) {
        int h_num = h_out - r + pad_h;
        if (h_num < 0 || h_num % stride_h != 0) continue;
        int h_in = h_num / stride_h;
        if (h_in < 0 || h_in >= H) continue;
        
        for (int s = 0; s < S; s++) {
            int w_num = w_out - s + pad_w;
            if (w_num < 0 || w_num % stride_w != 0) continue;
            int w_in = w_num / stride_w;
            if (w_in < 0 || w_in >= W) continue;
            
            int in_idx = ((n * H + h_in) * W + w_in) * C;
            int f_idx = (((0 * R + r) * S + s) * K + k_out);
            
            // Unroll C loop for better performance
            #pragma unroll 4
            for (int c = 0; c < C; c++) {
                float in_val = __half2float(input[in_idx + c]);
                float f_val = __half2float(filter[f_idx + c * R * S * K]);
                acc += in_val * f_val;
            }
        }
    }
    
    int out_idx = ((n * H_out + h_out) * W_out + w_out) * K + k_out;
    output[out_idx] = __float2half(acc);
}

// Most optimized: use shared memory for filter and better unrolling
#define BLOCK_W 16
#define BLOCK_H 16
#define BLOCK_K 16

__global__ void transposed_conv2d_fast_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ filter,
    __half* __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int H_out, int W_out)
{
    // Block: 16x16 threads, each computing 1 output pixel for 1 K value
    // Or: 8x8x8 threads, each computing partial sum
    
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int nt = blockDim.x * blockDim.y;
    
    const int w_out = blockIdx.x * BLOCK_W + threadIdx.x;
    const int h_out = blockIdx.y * BLOCK_H + threadIdx.y;
    const int n = blockIdx.z;
    
    if (w_out >= W_out || h_out >= H_out || n >= N) return;
    
    // Shared memory for accumulating across C dimension
    // Each thread stores partial sums for its K values
    
    // Process K in chunks
    for (int k_base = 0; k_base < K; k_base += BLOCK_K) {
        float acc[BLOCK_K]; // Each thread handles BLOCK_K values
        
        #pragma unroll
        for (int k = 0; k < BLOCK_K && k_base + k < K; k++) {
            acc[k] = 0.0f;
        }
        
        // Loop over R, S first (outer), then C (inner with shared filter)
        for (int r = 0; r < R; r++) {
            int h_num = h_out - r + pad_h;
            if (h_num < 0 || h_num % stride_h != 0) continue;
            int h_in = h_num / stride_h;
            if (h_in < 0 || h_in >= H) continue;
            
            for (int s = 0; s < S; s++) {
                int w_num = w_out - s + pad_w;
                if (w_num < 0 || w_num % stride_w != 0) continue;
                int w_in = w_num / stride_w;
                if (w_in < 0 || w_in >= W) continue;
                
                // Now iterate over C
                for (int c = 0; c < C; c++) {
                    float in_val = __half2float(input[((n * H + h_in) * W + w_in) * C + c]);
                    
                    // Load filter values for this c,r,s and all k in chunk
                    #pragma unroll
                    for (int k = 0; k < BLOCK_K && k_base + k < K; k++) {
                        float f_val = __half2float(filter[(((c * R + r) * S + s) * K + k_base + k)]);
                        acc[k] += in_val * f_val;
                    }
                }
            }
        }
        
        // Write output
        #pragma unroll
        for (int k = 0; k < BLOCK_K && k_base + k < K; k++) {
            int out_idx = ((n * H_out + h_out) * W_out + w_out) * K + k_base + k;
            output[out_idx] = __float2half(acc[k]);
        }
    }
}

// Final optimized kernel with proper vectorization and loop unrolling
template<int R_MAX, int S_MAX>
__global__ void transposed_conv2d_templated_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ filter,
    __half* __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int H_out, int W_out)
{
    // Each block computes a 16x16 tile of output spatial dimensions
    // Each thread computes 4 consecutive K values for coalesced writes
    
    const int tx = threadIdx.x;  // 0-15, handles W
    const int ty = threadIdx.y;  // 0-15, handles H  
    const int tz = threadIdx.z;  // 0-3, handles K/4
    
    const int w_out = blockIdx.x * 16 + tx;
    const int h_out = blockIdx.y * 16 + ty;
    const int n = blockIdx.z;
    
    if (w_out >= W_out || h_out >= H_out || n >= N) return;
    
    const int k_per_thread = 4;
    const int k_start = tz * k_per_thread;
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Precompute valid (r,s) pairs for this output pixel
    // This avoids conditionals in the inner loop
    
    #pragma unroll
    for (int r = 0; r < R_MAX && r < R; r++) {
        int h_num = h_out - r + pad_h;
        if (h_num < 0 || h_num % stride_h != 0) continue;
        int h_in = h_num / stride_h;
        if (h_in < 0 || h_in >= H) continue;
        
        #pragma unroll
        for (int s = 0; s < S_MAX && s < S; s++) {
            int w_num = w_out - s + pad_w;
            if (w_num < 0 || w_num % stride_w != 0) continue;
            int w_in = w_num / stride_w;
            if (w_in < 0 || w_in >= W) continue;
            
            int in_base = ((n * H + h_in) * W + w_in) * C;
            int f_base = (((0 * R + r) * S + s) * K);
            
            // Main accumulation loop over C
            // Use 4-way unrolling for better ILP
            int c = 0;
            for (; c + 3 < C; c += 4) {
                float in0 = __half2float(input[in_base + c + 0]);
                float in1 = __half2float(input[in_base + c + 1]);
                float in2 = __half2float(input[in_base + c + 2]);
                float in3 = __half2float(input[in_base + c + 3]);
                
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    int k_idx = k_start + k;
                    if (k_idx >= K) continue;
                    
                    acc[k] += in0 * __half2float(filter[f_base + (c + 0) * R * S * K + k_idx]);
                    acc[k] += in1 * __half2float(filter[f_base + (c + 1) * R * S * K + k_idx]);
                    acc[k] += in2 * __half2float(filter[f_base + (c + 2) * R * S * K + k_idx]);
                    acc[k] += in3 * __half2float(filter[f_base + (c + 3) * R * S * K + k_idx]);
                }
            }
            
            // Remainder
            for (; c < C; c++) {
                float in_val = __half2float(input[in_base + c]);
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    int k_idx = k_start + k;
                    if (k_idx >= K) continue;
                    acc[k] += in_val * __half2float(filter[f_base + c * R * S * K + k_idx]);
                }
            }
        }
    }
    
    // Coalesced write to output
    int out_base = ((n * H_out + h_out) * W_out + w_out) * K + k_start;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int k_idx = k_start + k;
        if (k_idx < K) {
            output[out_base + k] = __float2half(acc[k]);
        }
    }
}

// Wrapper to dispatch to appropriate template instance
__global__ void transposed_conv2d_dispatch_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ filter,
    __half* __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int H_out, int W_out)
{
    // Fallback generic kernel
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    
    const int w_out = blockIdx.x * 16 + tx;
    const int h_out = blockIdx.y * 16 + ty;
    const int n = blockIdx.z;
    
    if (w_out >= W_out || h_out >= H_out || n >= N) return;
    
    const int k_start = tz * 4;
    float acc[4] = {0.0f};
    
    for (int r = 0; r < R; r++) {
        int h_num = h_out - r + pad_h;
        if (h_num < 0 || h_num % stride_h != 0) continue;
        int h_in = h_num / stride_h;
        if (h_in < 0 || h_in >= H) continue;
        
        for (int s = 0; s < S; s++) {
            int w_num = w_out - s + pad_w;
            if (w_num < 0 || w_num % stride_w != 0) continue;
            int w_in = w_num / stride_w;
            if (w_in < 0 || w_in >= W) continue;
            
            for (int c = 0; c < C; c++) {
                float in_val = __half2float(input[((n * H + h_in) * W + w_in) * C + c]);
                for (int k = 0; k < 4 && k_start + k < K; k++) {
                    float f_val = __half2float(filter[(((c * R + r) * S + s) * K + k_start + k)]);
                    acc[k] += in_val * f_val;
                }
            }
        }
    }
    
    for (int k = 0; k < 4 && k_start + k < K; k++) {
        int out_idx = ((n * H_out + h_out) * W_out + w_out) * K + k_start + k;
        output[out_idx] = __float2half(acc[k]);
    }
}

// Main kernel with good performance characteristics
__global__ void transposed_conv2d_main_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ filter,
    __half* __restrict__ output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int H_out, int W_out)
{
    // Configuration: 16x16 threads for spatial, 4 for K dimension
    // Each thread computes 4 K values
    
    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15
    const int tk = threadIdx.z;  // 0-3, handles K/4
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bn = blockIdx.z;
    
    const int w_out = bx * 16 + tx;
    const int h_out = by * 16 + ty;
    const int n = bn;
    
    if (w_out >= W_out || h_out >= H_out || n >= N) return;
    
    const int K_PER_THREAD = 4;
    const int k0 = tk * K_PER_THREAD;
    
    // Accumulators
    float sum[K_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < K_PER_THREAD; i++) sum[i] = 0.0f;
    
    // Iterate over filter dimensions
    for (int r = 0; r < R; ++r) {
        const int h_f = h_out + pad_h - r;
        if (h_f < 0) continue;
        if (h_f % stride_h != 0) continue;
        const int h_in = h_f / stride_h;
        if (h_in >= H) continue;
        
        for (int s = 0; s < S; ++s) {
            const int w_f = w_out + pad_w - s;
            if (w_f < 0) continue;
            if (w_f % stride_w != 0) continue;
            const int w_in = w_f / stride_w;
            if (w_in >= W) continue;
            
            const int in_offset = ((n * H + h_in) * W + w_in) * C;
            
            // Inner loop over channels - manually unrolled for ILP
            int c = 0;
            #if 1
            // 4-way unrolled
            for (; c + 3 < C; c += 4) {
                float in0 = __half2float(input[in_offset + c + 0]);
                float in1 = __half2float(input[in_offset + c + 1]);
                float in2 = __half2float(input[in_offset + c + 2]);
                float in3 = __half2float(input[in_offset + c + 3]);
                
                #pragma unroll
                for (int k = 0; k < K_PER_THREAD; ++k) {
                    const int k_idx = k0 + k;
                    if (k_idx >= K) continue;
                    
                    const int f_off = (((c + 0) * R + r) * S + s) * K + k_idx;
                    sum[k] += in0 * __half2float(filter[f_off]);
                    sum[k] += in1 * __half2float(filter[f_off + R*S*K]);
                    sum[k] += in2 * __half2float(filter[f_off + 2*R*S*K]);
                    sum[k] += in3 * __half2float(filter[f_off + 3*R*S*K]);
                }
            }
            #endif
            
            // Remainder
            for (; c < C; ++c) {
                float in_val = __half2float(input[in_offset + c]);
                #pragma unroll
                for (int k = 0; k < K_PER_THREAD; ++k) {
                    const int k_idx = k0 + k;
                    if (k_idx >= K) continue;
                    const int f_idx = (((c * R + r) * S + s) * K + k_idx);
                    sum[k] += in_val * __half2float(filter[f_idx]);
                }
            }
        }
    }
    
    // Write output with coalesced access
    const int out_offset = ((n * H_out + h_out) * W_out + w_out) * K + k0;
    #pragma unroll
    for (int k = 0; k < K_PER_THREAD; ++k) {
        if (k0 + k < K) {
            output[out_offset + k] = __float2half(sum[k]);
        }
    }
}

cudaError_t TransposedConv2d(
    __half const *input, __half const *filter, __half *output,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w)
{
    // Calculate output dimensions
    int H_out = (H - 1) * stride_h + R - 2 * pad_h;
    int W_out = (W - 1) * stride_w + S - 2 * pad_w;
    
    // Ensure positive output dimensions
    if (H_out <= 0) H_out = 1;
    if (W_out <= 0) W_out = 1;
    
    // Grid and block configuration
    // Block: 16x16 spatial threads, 4 K-way threads = 1024 threads
    dim3 block(16, 16, 4);
    
    // Grid: cover output spatial dimensions and batch
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16, N);
    
    // Launch kernel
    transposed_conv2d_main_kernel<<<grid, block>>>(
        input, filter, output,
        N, C, H, W, K, R, S,
        pad_h, pad_w, stride_h, stride_w,
        H_out, W_out
    );
    
    return cudaGetLastError();
}
