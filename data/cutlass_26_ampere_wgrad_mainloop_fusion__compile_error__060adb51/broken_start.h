#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Conv2D Weight Gradient (wgrad) with fused per-channel scale + bias + ReLU activation.
// 
// Operation:
// 1. transformed[n][h][w][c] = max(0, scale[c] * input[n][h][w][c] + bias[c])
// 2. grad_weight[k][r][s][c] = sum_{n,p,q} grad_output[n][p][q][k] * transformed[n][h][w][c]
//    where h = p*stride_h + r - pad_h, w = q*stride_w + s - pad_w

// Tile dimensions for the kernel
#define TILE_K 32
#define TILE_R 3
#define TILE_S 3
#define TILE_C 32

// Use warp-level primitives for accumulation
#define WARP_SIZE 32

// Kernel: Compute wgrad with fused scale+bias+ReLU on input
__global__ void Conv2dWgradScaleBiasKernel(
    const __half* __restrict__ input,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ scale,
    const __half* __restrict__ bias,
    float* __restrict__ grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q)
{
    // Block dimensions: each block handles a tile of output (K x R x S x C)
    // We parallelize over K, R*S, and C dimensions
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Each warp handles a portion of the C dimension
    const int c_tile_start = blockIdx.z * TILE_C;
    const int c_local = lane_id;  // Each thread in warp handles one C
    
    // K and spatial (R*S) are split across blocks
    const int k_start = blockIdx.x * TILE_K;
    const int rs_start = blockIdx.y * (TILE_R * TILE_S);
    
    // Determine which (r,s) this thread block handles
    const int total_rs = R * S;
    
    // Shared memory for transformed input and grad_output
    // We'll process N, P, Q in batches
    
    // Each thread accumulates grad_weight for its assigned (k, r, s, c)
    float accum[TILE_K] = {0.0f};  // Accumulate for different k values
    
    // Preload scale and bias for this C tile into registers
    __half scale_val[TILE_C];
    __half bias_val[TILE_C];
    if (c_tile_start + c_local < C && c_local < TILE_C) {
        scale_val[c_local] = scale[c_tile_start + c_local];
        bias_val[c_local] = bias[c_tile_start + c_local];
    }
    
    // Iterate over N, P, Q to accumulate the gradient
    for (int n = 0; n < N; ++n) {
        for (int p = 0; p < P; ++p) {
            for (int q = 0; q < Q; ++q) {
                // For each (r,s) in this block's tile
                for (int rs = 0; rs < TILE_R * TILE_S; ++rs) {
                    int rs_idx = rs_start + rs;
                    if (rs_idx >= total_rs) continue;
                    
                    int r = rs_idx / S;
                    int s = rs_idx % S;
                    
                    // Compute corresponding (h, w) in input
                    int h = p * stride_h + r - pad_h;
                    int w = q * stride_w + s - pad_w;
                    
                    // Check bounds
                    if (h < 0 || h >= H || w < 0 || w >= W) continue;
                    
                    // Load and transform input: max(0, scale[c] * input + bias[c])
                    int input_idx = ((n * H + h) * W + w) * C + (c_tile_start + c_local);
                    float transformed_val = 0.0f;
                    
                    if (c_tile_start + c_local < C && c_local < TILE_C) {
                        float in_val = __half2float(input[input_idx]);
                        float scaled = __half2float(scale_val[c_local]) * in_val + 
                                      __half2float(bias_val[c_local]);
                        transformed_val = (scaled > 0.0f) ? scaled : 0.0f;  // ReLU
                    }
                    
                    // Load grad_output for different k values
                    for (int kk = 0; kk < TILE_K; ++kk) {
                        int k_idx = k_start + kk;
                        if (k_idx >= K) continue;
                        
                        int grad_out_idx = ((n * P + p) * Q + q) * K + k_idx;
                        float go_val = __half2float(grad_output[grad_out_idx]);
                        
                        accum[kk] += go_val * transformed_val;
                    }
                }
            }
        }
    }
    
    // Write out results to grad_weight
    // grad_weight layout: K x R x S x C
    for (int rs = 0; rs < TILE_R * TILE_S; ++rs) {
        int rs_idx = rs_start + rs;
        if (rs_idx >= total_rs) continue;
        
        int r = rs_idx / S;
        int s = rs_idx % S;
        
        for (int kk = 0; kk < TILE_K; ++kk) {
            int k_idx = k_start + kk;
            if (k_idx >= K) continue;
            
            if (c_tile_start + c_local < C && c_local < TILE_C) {
                int gw_idx = ((k_idx * R + r) * S + s) * C + (c_tile_start + c_local);
                atomicAdd(&grad_weight[gw_idx], accum[kk]);
            }
        }
    }
}

// Optimized kernel using shared memory for better memory access patterns
__global__ void Conv2dWgradScaleBiasKernelOptimized(
    const __half* __restrict__ input,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ scale,
    const __half* __restrict__ bias,
    float* __restrict__ grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q)
{
    // Shared memory layout:
    // smem_transformed: holds transformed input for tile of (N, P, Q, C_tile)
    // smem_grad_out: holds grad_output for tile of (N, P, Q, K_tile)
    
    // Each block handles: K_tile x C_tile for all R*S
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Tile dimensions
    const int K_TILE = 16;
    const int C_TILE = 16;
    const int NPQ_TILE = 4;  // Process 4 (n,p,q) at a time
    
    // Block indexing
    const int k_block = blockIdx.x * K_TILE;
    const int c_block = blockIdx.y * C_TILE;
    const int rs_block = blockIdx.z;  // Each z-block handles one (r,s) pair
    
    if (rs_block >= R * S) return;
    const int r = rs_block / S;
    const int s = rs_block % S;
    
    // Registers for accumulation
    float accum[K_TILE][C_TILE];
    #pragma unroll
    for (int i = 0; i < K_TILE; ++i) {
        #pragma unroll
        for (int j = 0; j < C_TILE; ++j) {
            accum[i][j] = 0.0f;
        }
    }
    
    // Preload scale and bias for C tile
    __half scale_reg[C_TILE];
    __half bias_reg[C_TILE];
    #pragma unroll
    for (int cc = 0; cc < C_TILE; ++cc) {
        int c_idx = c_block + cc;
        if (c_idx < C) {
            scale_reg[cc] = scale[c_idx];
            bias_reg[cc] = bias[c_idx];
        }
    }
    
    // Shared memory
    extern __shared__ char smem[];
    float* smem_transformed = (float*)smem;  // [NPQ_TILE][C_TILE]
    float* smem_grad_out = (float*)&smem_transformed[NPQ_TILE * C_TILE];  // [NPQ_TILE][K_TILE]
    
    // Iterate over N, P, Q in tiles
    for (int npq_base = 0; npq_base < N * P * Q; npq_base += NPQ_TILE) {
        // Load transformed input into shared memory
        // Each thread loads some elements
        for (int idx = tid; idx < NPQ_TILE * C_TILE; idx += block_size) {
            int npq_idx = idx / C_TILE;
            int cc = idx % C_TILE;
            
            int global_npq = npq_base + npq_idx;
            int n = global_npq / (P * Q);
            int pq_rem = global_npq % (P * Q);
            int p = pq_rem / Q;
            int q = pq_rem % Q;
            
            float val = 0.0f;
            if (n < N && c_block + cc < C) {
                int h = p * stride_h + r - pad_h;
                int w = q * stride_w + s - pad_w;
                
                if (h >= 0 && h < H && w >= 0 && w < W) {
                    int input_idx = ((n * H + h) * W + w) * C + (c_block + cc);
                    float in_val = __half2float(input[input_idx]);
                    float scaled = __half2float(scale_reg[cc]) * in_val + 
                                  __half2float(bias_reg[cc]);
                    val = (scaled > 0.0f) ? scaled : 0.0f;
                }
            }
            smem_transformed[npq_idx * C_TILE + cc] = val;
        }
        
        // Load grad_output into shared memory
        for (int idx = tid; idx < NPQ_TILE * K_TILE; idx += block_size) {
            int npq_idx = idx / K_TILE;
            int kk = idx % K_TILE;
            
            int global_npq = npq_base + npq_idx;
            int n = global_npq / (P * Q);
            int pq_rem = global_npq % (P * Q);
            int p = pq_rem / Q;
            int q = pq_rem % Q;
            
            float val = 0.0f;
            if (n < N && k_block + kk < K) {
                int grad_out_idx = ((n * P + p) * Q + q) * K + (k_block + kk);
                val = __half2float(grad_output[grad_out_idx]);
            }
            smem_grad_out[npq_idx * K_TILE + kk] = val;
        }
        
        __syncthreads();
        
        // Compute outer product
        #pragma unroll
        for (int npq = 0; npq < NPQ_TILE; ++npq) {
            #pragma unroll
            for (int kk = 0; kk < K_TILE; ++kk) {
                float go_val = smem_grad_out[npq * K_TILE + kk];
                #pragma unroll
                for (int cc = 0; cc < C_TILE; ++cc) {
                    accum[kk][cc] += go_val * smem_transformed[npq * C_TILE + cc];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results using atomicAdd
    // Each thread writes some elements
    for (int idx = tid; idx < K_TILE * C_TILE; idx += block_size) {
        int kk = idx / C_TILE;
        int cc = idx % C_TILE;
        
        int k_idx = k_block + kk;
        int c_idx = c_block + cc;
        
        if (k_idx < K && c_idx < C) {
            int gw_idx = ((k_idx * R + r) * S + s) * C + c_idx;
            atomicAdd(&grad_weight[gw_idx], accum[kk][cc]);
        }
    }
}

// Even more optimized kernel - process multiple (r,s) per block
__global__ void Conv2dWgradScaleBiasKernelV2(
    const __half* __restrict__ input,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ scale,
    const __half* __restrict__ bias,
    float* __restrict__ grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    // Each warp handles a subset of the C dimension
    // Each block handles a K-tile and multiple (r,s) positions
    
    const int K_TILE = 32;
    const int C_PER_WARP = 32;
    
    const int k_block = blockIdx.x * K_TILE;
    const int c_block = blockIdx.y * C_PER_WARP * num_warps + warp_id * C_PER_WARP;
    const int rs_start = blockIdx.z * 4;  // Each z-block handles 4 (r,s) pairs
    
    if (c_block >= C) return;
    
    // Accumulators for this warp
    float accum[4][K_TILE][C_PER_WARP];  // [rs][k][c]
    #pragma unroll
    for (int rs = 0; rs < 4; ++rs) {
        #pragma unroll
        for (int k = 0; k < K_TILE; ++k) {
            #pragma unroll
            for (int c = 0; c < C_PER_WARP; ++c) {
                accum[rs][k][c] = 0.0f;
            }
        }
    }
    
    // Load scale and bias
    __half scale_reg[C_PER_WARP];
    __half bias_reg[C_PER_WARP];
    #pragma unroll
    for (int c = 0; c < C_PER_WARP; ++c) {
        int c_idx = c_block + c;
        if (c_idx < C && lane_id == 0) {
            scale_reg[c] = scale[c_idx];
            bias_reg[c] = bias[c_idx];
        }
    }
    // Broadcast within warp
    #pragma unroll
    for (int c = 0; c < C_PER_WARP; ++c) {
        unsigned int s_raw, b_raw;
        s_raw = __half_raw(scale_reg[c]).x;
        b_raw = __half_raw(bias_reg[c]).x;
        s_raw = __shfl_sync(0xffffffff, s_raw, 0);
        b_raw = __shfl_sync(0xffffffff, b_raw, 0);
        scale_reg[c] = __half_raw((unsigned short)s_raw);
        bias_reg[c] = __half_raw((unsigned short)b_raw);
    }
    
    // Iterate over N, P, Q
    for (int n = 0; n < N; ++n) {
        for (int p = 0; p < P; ++p) {
            for (int q_base = 0; q_base < Q; q_base += 32) {
                int q = q_base + lane_id;
                
                // Load grad_output for all K in K_TILE
                float go_vals[K_TILE];
                #pragma unroll
                for (int k = 0; k < K_TILE; ++k) {
                    go_vals[k] = 0.0f;
                    int k_idx = k_block + k;
                    if (q < Q && k_idx < K) {
                        int go_idx = ((n * P + p) * Q + q) * K + k_idx;
                        go_vals[k] = __half2float(grad_output[go_idx]);
                    }
                }
                
                // For each (r,s) in this block's tile
                #pragma unroll
                for (int rs = 0; rs < 4; ++rs) {
                    int rs_idx = rs_start + rs;
                    if (rs_idx >= R * S) continue;
                    
                    int r = rs_idx / S;
                    int s = rs_idx % S;
                    
                    int h = p * stride_h + r - pad_h;
                    int w = q * stride_w + s - pad_w;
                    
                    // Load and transform input for this C warp
                    float in_vals[C_PER_WARP];
                    #pragma unroll
                    for (int c = 0; c < C_PER_WARP; ++c) {
                        in_vals[c] = 0.0f;
                        int c_idx = c_block + c;
                        if (q < Q && c_idx < C && h >= 0 && h < H && w >= 0 && w < W) {
                            int in_idx = ((n * H + h) * W + w) * C + c_idx;
                            float in_val = __half2float(input[in_idx]);
                            float scaled = __half2float(scale_reg[c]) * in_val + 
                                          __half2float(bias_reg[c]);
                            in_vals[c] = (scaled > 0.0f) ? scaled : 0.0f;
                        }
                    }
                    
                    // Accumulate outer product
                    #pragma unroll
                    for (int k = 0; k < K_TILE; ++k) {
                        #pragma unroll
                        for (int c = 0; c < C_PER_WARP; ++c) {
                            accum[rs][k][c] += go_vals[k] * in_vals[c];
                        }
                    }
                }
            }
        }
    }
    
    // Write results with atomicAdd
    // Use warp shuffle to reduce first, then atomic add
    #pragma unroll
    for (int rs = 0; rs < 4; ++rs) {
        int rs_idx = rs_start + rs;
        if (rs_idx >= R * S) continue;
        int r = rs_idx / S;
        int s = rs_idx % S;
        
        #pragma unroll
        for (int k = 0; k < K_TILE; ++k) {
            #pragma unroll
            for (int c = 0; c < C_PER_WARP; ++c) {
                int k_idx = k_block + k;
                int c_idx = c_block + c;
                if (k_idx < K && c_idx < C) {
                    // Warp-level reduction
                    float val = accum[rs][k][c];
                    // Reduce across warp using shuffle
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        val += __shfl_down_sync(0xffffffff, val, offset);
                    }
                    
                    if (lane_id == 0) {
                        int gw_idx = ((k_idx * R + r) * S + s) * C + c_idx;
                        atomicAdd(&grad_weight[gw_idx], val);
                    }
                }
            }
        }
    }
}

// Simple but correct baseline kernel
__global__ void Conv2dWgradScaleBiasKernelSimple(
    const __half* __restrict__ input,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ scale,
    const __half* __restrict__ bias,
    float* __restrict__ grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q)
{
    // Each thread computes one element of grad_weight: grad_weight[k][r][s][c]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int total_elements = K * R * S * C;
    if (idx >= total_elements) return;
    
    // Decode index
    int c = idx % C;
    int rs = (idx / C) % (R * S);
    int r = rs / S;
    int s = rs % S;
    int k = idx / (R * S * C);
    
    float accum = 0.0f;
    
    __half sc = scale[c];
    __half bi = bias[c];
    
    for (int n = 0; n < N; ++n) {
        for (int p = 0; p < P; ++p) {
            for (int q = 0; q < Q; ++q) {
                int h = p * stride_h + r - pad_h;
                int w = q * stride_w + s - pad_w;
                
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                
                // Load and transform input
                int in_idx = ((n * H + h) * W + w) * C + c;
                float in_val = __half2float(input[in_idx]);
                float transformed = __half2float(sc) * in_val + __half2float(bi);
                if (transformed < 0.0f) transformed = 0.0f;  // ReLU
                
                // Load grad_output
                int go_idx = ((n * P + p) * Q + q) * K + k;
                float go_val = __half2float(grad_output[go_idx]);
                
                accum += go_val * transformed;
            }
        }
    }
    
    // Write result
    grad_weight[idx] = accum;
}

// Medium optimized kernel - better thread utilization
__global__ void Conv2dWgradScaleBiasKernelMedium(
    const __half* __restrict__ input,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ scale,
    const __half* __restrict__ bias,
    float* __restrict__ grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q)
{
    // Block structure: each block handles a (K_TILE x C_TILE) tile for all R*S
    // Threads within block collaborate on the N*P*Q reduction
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    const int K_TILE = 8;
    const int C_TILE = 8;
    
    const int k_block = blockIdx.x * K_TILE;
    const int c_block = blockIdx.y * C_TILE;
    
    // Each thread handles a subset of (r,s) and accumulates over N,P,Q
    // Distribute R*S across threads in block
    const int total_rs = R * S;
    
    // Accumulators: each thread handles some (r,s) pairs
    // We'll have each thread loop over its assigned (r,s) values
    
    for (int rs_idx = tid; rs_idx < total_rs; rs_idx += block_size) {
        int r = rs_idx / S;
        int s = rs_idx % S;
        
        float accum[K_TILE][C_TILE];
        #pragma unroll
        for (int kk = 0; kk < K_TILE; ++kk) {
            #pragma unroll
            for (int cc = 0; cc < C_TILE; ++cc) {
                accum[kk][cc] = 0.0f;
            }
        }
        
        // Preload scale/bias for C tile
        float scale_f[C_TILE];
        float bias_f[C_TILE];
        #pragma unroll
        for (int cc = 0; cc < C_TILE; ++cc) {
            int c_idx = c_block + cc;
            if (c_idx < C) {
                scale_f[cc] = __half2float(scale[c_idx]);
                bias_f[cc] = __half2float(bias[c_idx]);
            }
        }
        
        // Accumulate over N, P, Q
        for (int n = 0; n < N; ++n) {
            for (int p = 0; p < P; ++p) {
                for (int q = 0; q < Q; ++q) {
                    int h = p * stride_h + r - pad_h;
                    int w = q * stride_w + s - pad_w;
                    
                    if (h < 0 || h >= H || w < 0 || w >= W) continue;
                    
                    // Load grad_output for K tile
                    float go_vals[K_TILE];
                    #pragma unroll
                    for (int kk = 0; kk < K_TILE; ++kk) {
                        int k_idx = k_block + kk;
                        go_vals[kk] = 0.0f;
                        if (k_idx < K) {
                            int go_idx = ((n * P + p) * Q + q) * K + k_idx;
                            go_vals[kk] = __half2float(grad_output[go_idx]);
                        }
                    }
                    
                    // Load and transform input for C tile
                    float in_vals[C_TILE];
                    #pragma unroll
                    for (int cc = 0; cc < C_TILE; ++cc) {
                        int c_idx = c_block + cc;
                        in_vals[cc] = 0.0f;
                        if (c_idx < C) {
                            int in_idx = ((n * H + h) * W + w) * C + c_idx;
                            float in_val = __half2float(input[in_idx]);
                            float transformed = scale_f[cc] * in_val + bias_f[cc];
                            in_vals[cc] = (transformed > 0.0f) ? transformed : 0.0f;
                        }
                    }
                    
                    // Outer product
                    #pragma unroll
                    for (int kk = 0; kk < K_TILE; ++kk) {
                        #pragma unroll
                        for (int cc = 0; cc < C_TILE; ++cc) {
                            accum[kk][cc] += go_vals[kk] * in_vals[cc];
                        }
                    }
                }
            }
        }
        
        // Write results
        #pragma unroll
        for (int kk = 0; kk < K_TILE; ++kk) {
            #pragma unroll
            for (int cc = 0; cc < C_TILE; ++cc) {
                int k_idx = k_block + kk;
                int c_idx = c_block + cc;
                if (k_idx < K && c_idx < C) {
                    int gw_idx = ((k_idx * R + r) * S + s) * C + c_idx;
                    atomicAdd(&grad_weight[gw_idx], accum[kk][cc]);
                }
            }
        }
    }
}

// High performance kernel with good memory coalescing
__global__ void Conv2dWgradScaleBiasKernelFast(
    const __half* __restrict__ input,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ scale,
    const __half* __restrict__ bias,
    float* __restrict__ grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int P, int Q)
{
    // Configuration: each block handles (K_TILE=16, C_TILE=16) for one (r,s)
    // Grid: (K/16, C/16, R*S)
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    const int K_TILE = 16;
    const int C_TILE = 16;
    const int NPQ_PER_ITER = 4;  // Process 4 (n,p,q) per iteration
    
    const int k_block = blockIdx.x * K_TILE;
    const int c_block = blockIdx.y * C_TILE;
    const int rs_idx = blockIdx.z;
    
    if (rs_idx >= R * S) return;
    const int r = rs_idx / S;
    const int s = rs_idx % S;
    
    // Each warp handles a subset of the reduction
    // Accumulators: [K_TILE][C_TILE]
    float accum[K_TILE][C_TILE];
    #pragma unroll
    for (int kk = 0; kk < K_TILE; ++kk) {
        #pragma unroll
        for (int cc = 0; cc < C_TILE; ++cc) {
            accum[kk][cc] = 0.0f;
        }
    }
    
    // Preload scale and bias
    float scale_f[C_TILE];
    float bias_f[C_TILE];
    #pragma unroll
    for (int cc = 0; cc < C_TILE; ++cc) {
        int c_idx = c_block + cc;
        scale_f[cc] = (c_idx < C) ? __half2float(scale[c_idx]) : 0.0f;
        bias_f[cc] = (c_idx < C) ? __half2float(bias[c_idx]) : 0.0f;
    }
    
    // Shared memory for grad_output and transformed input
    // Layout: smem_go[NPQ_PER_ITER][K_TILE], smem_in[NPQ_PER_ITER][C_TILE]
    extern __shared__ char smem[];
    float* smem_go = (float*)smem;  // [num_warps][NPQ_PER_ITER][K_TILE]
    float* smem_in = (float*)&smem_go[num_warps * NPQ_PER_ITER * K_TILE];  // [num_warps][NPQ_PER_ITER][C_TILE]
    
    float* my_smem_go = &smem_go[warp_id * NPQ_PER_ITER * K_TILE];
    float* my_smem_in = &smem_in[warp_id * NPQ_PER_ITER * C_TILE];
    
    // Total NPQ
    const int total_npq = N * P * Q;
    
    // Each warp processes a subset of NPQ
    for (int npq_base = warp_id * NPQ_PER_ITER; npq_base < total_npq; npq_base += num_warps * NPQ_PER_ITER) {
        // Load data into shared memory
        // Each thread in warp loads some elements
        #pragma unroll
        for (int i = 0; i < NPQ_PER_ITER; ++i) {
            int npq_idx = npq_base + i;
            
            // Decode NPQ
            int n = npq_idx / (P * Q);
            int pq_rem = npq_idx % (P * Q);
            int p = pq_rem / Q;
            int q = pq_rem % Q;
            
            // Compute h, w
            int h = p * stride_h + r - pad_h;
            int w = q * stride_w + s - pad_w;
            
            bool valid = (n < N && h >= 0 && h < H && w >= 0 && w < W);
            
            // Load grad_output for K_TILE
            #pragma unroll
            for (int kk = lane_id; kk < K_TILE; kk += 32) {
                float val = 0.0f;
                if (valid && k_block + kk < K) {
                    int go_idx = ((n * P + p) * Q + q) * K + (k_block + kk);
                    val = __half2float(grad_output[go_idx]);
                }
                my_smem_go[i * K_TILE + kk] = val;
            }
            
            // Load and transform input for C_TILE
            #pragma unroll
            for (int cc = lane_id; cc < C_TILE; cc += 32) {
                float val = 0.0f;
                if (valid && c_block + cc < C) {
                    int in_idx = ((n * H + h) * W + w) * C + (c_block + cc);
                    float in_val = __half2float(input[in_idx]);
                    float transformed = scale_f[cc] * in_val + bias_f[cc];
                    val = (transformed > 0.0f) ? transformed : 0.0f;
                }
                my_smem_in[i * C_TILE + cc] = val;
            }
        }
        
        // Compute outer product
        #pragma unroll
        for (int i = 0; i < NPQ_PER_ITER; ++i) {
            #pragma unroll
            for (int kk = 0; kk < K_TILE; ++kk) {
                float go_val = my_smem_go[i * K_TILE + kk];
                #pragma unroll
                for (int cc = 0; cc < C_TILE; ++cc) {
                    accum[kk][cc] += go_val * my_smem_in[i * C_TILE + cc];
                }
            }
        }
    }
    
    // Warp-level reduction (accum is same across all threads in warp, so no need)
    // Actually each warp computed partial sum, need to reduce across warps
    
    // Store accum to shared memory and reduce
    // Use shared memory for inter-warp reduction
    float* smem_reduce = (float*)smem;  // Reuse smem
    // Each warp writes its accum
    #pragma unroll
    for (int kk = 0; kk < K_TILE; ++kk) {
        #pragma unroll
        for (int cc = lane_id; cc < C_TILE; cc += 32) {
            smem_reduce[warp_id * K_TILE * C_TILE + kk * C_TILE + cc] = accum[kk][cc];
        }
    }
    
    __syncthreads();
    
    // Warp 0 reduces and writes output
    if (warp_id == 0) {
        #pragma unroll
        for (int kk = 0; kk < K_TILE; ++kk) {
            #pragma unroll
            for (int cc = lane_id; cc < C_TILE; cc += 32) {
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < num_warps; ++w) {
                    sum += smem_reduce[w * K_TILE * C_TILE + kk * C_TILE + cc];
                }
                
                int k_idx = k_block + kk;
                int c_idx = c_block + cc;
                if (k_idx < K && c_idx < C) {
                    int gw_idx = ((k_idx * R + r) * S + s) * C + c_idx;
                    grad_weight[gw_idx] = sum;
                }
            }
        }
    }
}

// Main entry point
cudaError_t Conv2dWgradScaleBias(
    __half const *input, __half const *grad_output,
    __half const *scale, __half const *bias,
    float *grad_weight,
    int N, int C, int H, int W, int K, int R, int S,
    int pad_h, int pad_w, int stride_h, int stride_w)
{
    // Compute output dimensions
    int P = (H + 2 * pad_h - R) / stride_h + 1;
    int Q = (W + 2 * pad_w - S) / stride_w + 1;
    
    // Zero grad_weight
    size_t grad_weight_size = K * R * S * C * sizeof(float);
    cudaMemset(grad_weight, 0, grad_weight_size);
    
    // Choose kernel based on problem size
    // For small problems, use simple kernel
    // For larger problems, use optimized kernel
    
    // Check if dimensions are reasonable for fast kernel
    bool use_fast = (K >= 16 && C >= 16 && R * S <= 64);
    
    if (use_fast) {
        // Fast kernel: grid over K, C, R*S
        const int K_TILE = 16;
        const int C_TILE = 16;
        
        int grid_k = (K + K_TILE - 1) / K_TILE;
        int grid_c = (C + C_TILE - 1) / C_TILE;
        int grid_rs = R * S;
        
        dim3 grid(grid_k, grid_c, grid_rs);
        dim3 block(128);  // 4 warps
        
        size_t smem_size = 4 * 16 * sizeof(float) + 4 * 16 * sizeof(float);  // Conservative estimate
        smem_size = 8192;  // 8KB
        
        Conv2dWgradScaleBiasKernelFast<<<grid, block, smem_size>>>(
            input, grad_output, scale, bias, grad_weight,
            N, C, H, W, K, R, S,
            pad_h, pad_w, stride_h, stride_w,
            P, Q);
    } else {
        // Simple kernel for small or irregular sizes
        int total_output = K * R * S * C;
        int block_size = 256;
        int grid_size = (total_output + block_size - 1) / block_size;
        
        Conv2dWgradScaleBiasKernelSimple<<<grid_size, block_size>>>(
            input, grad_output, scale, bias, grad_weight,
            N, C, H, W, K, R, S,
            pad_h, pad_w, stride_h, stride_w,
            P, Q);
    }
    
    return cudaGetLastError();
}
