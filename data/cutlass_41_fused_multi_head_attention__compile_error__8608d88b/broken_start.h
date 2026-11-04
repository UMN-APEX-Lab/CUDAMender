#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

// Fused Multi-Head Attention kernel
// Uses warp-level primitives for efficient computation
// Each block handles one (batch, head) pair, processing multiple query rows

// Constants for tile sizes
constexpr int WARP_SIZE = 32;
constexpr int TILE_Q = 64;   // Query rows per block
constexpr int TILE_KV = 64;  // Key/value columns per block (for QK matmul)
constexpr int TILE_V = 64;   // Value dimension tile

// Use float for accumulation
using acc_t = float;

// Helper to convert __half to float
__device__ __forceinline__ float to_float(__half h) {
    return __half2float(h);
}

// Helper to convert float to __half
__device__ __forceinline__ __half to_half(float f) {
    return __float2half(f);
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level max reduction
__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block-level max reduction (using shared memory)
__device__ float block_max(float val, float* shared_mem) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    
    // Warp-level max
    val = warp_max(val);
    
    // Store to shared memory
    if (lane == 0) shared_mem[warp] = val;
    __syncthreads();
    
    // Final reduction across warps
    if (warp == 0) {
        val = (lane < blockDim.x / WARP_SIZE) ? shared_mem[lane] : -INFINITY;
        val = warp_max(val);
        if (lane == 0) shared_mem[0] = val;
    }
    __syncthreads();
    
    return shared_mem[0];
}

// Block-level sum reduction (using shared memory)
__device__ float block_sum(float val, float* shared_mem) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    
    // Warp-level sum
    val = warp_sum(val);
    
    // Store to shared memory
    if (lane == 0) shared_mem[warp] = val;
    __syncthreads();
    
    // Final reduction across warps
    if (warp == 0) {
        val = (lane < blockDim.x / WARP_SIZE) ? shared_mem[lane] : 0.0f;
        val = warp_sum(val);
        if (lane == 0) shared_mem[0] = val;
    }
    __syncthreads();
    
    return shared_mem[0];
}

// Main FMHA kernel
// Each block processes TILE_Q query rows for one (batch, head)
__global__ void fmha_kernel(
    int batch_size, int seq_q, int seq_kv,
    int num_heads, int head_dim, int head_dim_v,
    __half const *Q, __half const *K, __half const *V, __half *O,
    int q_strideB, int q_strideM, int q_strideH,
    int k_strideB, int k_strideM, int k_strideH,
    int v_strideB, int v_strideM, int v_strideH,
    int o_strideM,
    bool causal) {
    
    // Block indices
    int bh_idx = blockIdx.x;  // batch * num_heads + head
    int batch = bh_idx / num_heads;
    int head = bh_idx % num_heads;
    int q_row_start = blockIdx.y * TILE_Q;
    
    // Thread indices
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x / WARP_SIZE;
    
    // Shared memory layout:
    // - Q tile: TILE_Q * head_dim
    // - K tile: TILE_KV * head_dim  
    // - V tile: TILE_KV * head_dim_v
    // - S tile: TILE_Q * TILE_KV (scores)
    // - Reduction workspace: num_warps floats
    
    extern __shared__ char smem[];
    float* red_smem = (float*)smem;  // size: num_warps
    
    // Pointers to current batch/head
    const __half* Q_ptr = Q + batch * q_strideB + head * q_strideH;
    const __half* K_ptr = K + batch * k_strideB + head * k_strideH;
    const __half* V_ptr = V + batch * v_strideB + head * v_strideH;
    __half* O_ptr = O + batch * (seq_q * o_strideM) + head * head_dim_v;
    
    // Scale factor
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Each thread processes elements
    // We use a simple approach: each thread computes partial dot products
    
    // For each query row assigned to this block
    for (int q_local = 0; q_local < TILE_Q; q_local++) {
        int q_row = q_row_start + q_local;
        if (q_row >= seq_q) break;
        
        // Online softmax with tiling over KV sequence
        // We process KV in chunks, maintaining running max and sum
        
        float row_max = -INFINITY;
        float row_sum = 0.0f;
        
        // First pass: compute QK^T and find max (with causal masking)
        for (int kv_start = 0; kv_start < seq_kv; kv_start += TILE_KV) {
            // Compute Q[row,:] dot K[kv,:] for this tile
            // Each thread computes partial sums over head_dim
            
            float local_max = -INFINITY;
            
            // Process this KV tile
            for (int kv_idx = kv_start + tid; kv_idx < kv_start + TILE_KV && kv_idx < seq_kv; kv_idx += blockDim.x) {
                // Apply causal mask
                if (causal && kv_idx > q_row) {
                    continue;
                }
                
                // Compute dot product Q[q_row, :] · K[kv_idx, :]
                float dot = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < head_dim; d++) {
                    float q_val = to_float(Q_ptr[q_row * q_strideM + d]);
                    float k_val = to_float(K_ptr[kv_idx * k_strideM + d]);
                    dot += q_val * k_val;
                }
                dot *= scale;
                
                local_max = fmaxf(local_max, dot);
            }
            
            // Reduce to find max in this tile
            float tile_max = block_max(local_max, red_smem);
            row_max = fmaxf(row_max, tile_max);
        }
        
        // Broadcast row_max to all threads
        if (tid == 0) red_smem[0] = row_max;
        __syncthreads();
        row_max = red_smem[0];
        __syncthreads();
        
        // Second pass: compute softmax probabilities and weighted sum with V
        // We need to accumulate: sum_j exp(S[q,j] - m) * V[j,:]
        
        // Accumulators for output
        float* acc = (float*)(smem + num_warps * sizeof(float));
        for (int d = 0; d < head_dim_v; d++) {
            acc[tid * head_dim_v + d] = 0.0f;  // This won't fit, use registers instead
        }
        
        // Use registers for accumulation - each thread handles a subset of output dimensions
        float out_acc[8];  // Assume head_dim_v <= 64, each thread handles up to 8 elements
        #pragma unroll
        for (int i = 0; i < 8; i++) out_acc[i] = 0.0f;
        
        // Actually, let's use a simpler approach: process V dimensions in tiles too
        
        // Reset row_sum
        row_sum = 0.0f;
        
        for (int kv_start = 0; kv_start < seq_kv; kv_start += TILE_KV) {
            // Compute softmax values for this KV tile
            for (int kv_idx = kv_start + tid; kv_idx < kv_start + TILE_KV && kv_idx < seq_kv; kv_idx += blockDim.x) {
                if (causal && kv_idx > q_row) continue;
                
                // Compute dot product
                float dot = 0.0f;
                #pragma unroll 4
                for (int d = 0; d < head_dim; d++) {
                    float q_val = to_float(Q_ptr[q_row * q_strideM + d]);
                    float k_val = to_float(K_ptr[kv_idx * k_strideM + d]);
                    dot += q_val * k_val;
                }
                dot *= scale;
                
                float exp_val = expf(dot - row_max);
                row_sum += exp_val;
                
                // Accumulate weighted V
                // Each thread handles different V dimensions
                for (int d = lane; d < head_dim_v; d += WARP_SIZE) {
                    float v_val = to_float(V_ptr[kv_idx * v_strideM + d]);
                    out_acc[d / WARP_SIZE] += exp_val * v_val;
                }
            }
        }
        
        // Reduce row_sum across block
        row_sum = block_sum(row_sum, red_smem);
        
        // Normalize and write output
        float inv_sum = 1.0f / row_sum;
        
        // Warp-level reduction of accumulators
        for (int d = 0; d < head_dim_v; d += WARP_SIZE) {
            int idx = d / WARP_SIZE;
            if (d + lane < head_dim_v) {
                float val = out_acc[idx];
                // Reduce across warps using shared memory
                if (warp == 0) {
                    // Collect from all warps
                    val = warp_sum(val);
                    if (lane == 0) {
                        O_ptr[q_row * o_strideM + d] = to_half(val * inv_sum);
                    }
                }
            }
        }
    }
}

// Optimized FMHA kernel using shared memory for Q, K, V tiles
__global__ void fmha_kernel_optimized(
    int batch_size, int seq_q, int seq_kv,
    int num_heads, int head_dim, int head_dim_v,
    __half const *Q, __half const *K, __half const *V, __half *O,
    int q_strideB, int q_strideM, int q_strideH,
    int k_strideB, int k_strideM, int k_strideH,
    int v_strideB, int v_strideM, int v_strideH,
    int o_strideM,
    bool causal) {
    
    // Each block handles one query row (or small group) for better occupancy
    // Using warp-level matrix multiplication approach
    
    int bh_idx = blockIdx.x;
    int batch = bh_idx / num_heads;
    int head = bh_idx % num_heads;
    int q_row = blockIdx.y;
    
    if (q_row >= seq_q) return;
    
    int tid = threadIdx.x;
    int lane = tid & 31;
    
    // Pointers
    const __half* Q_ptr = Q + batch * q_strideB + head * q_strideH;
    const __half* K_ptr = K + batch * k_strideB + head * k_strideH;
    const __half* V_ptr = V + batch * v_strideB + head * v_strideH;
    __half* O_ptr = O + batch * (seq_q * o_strideM) + head * head_dim_v;
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Shared memory for reduction
    extern __shared__ float smem[];
    
    // Step 1: Compute QK^T for this query row, find max
    float row_max = -INFINITY;
    
    // Each thread handles multiple KV positions
    for (int kv = tid; kv < seq_kv; kv += blockDim.x) {
        if (causal && kv > q_row) continue;
        
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            float q_val = to_float(Q_ptr[q_row * q_strideM + d]);
            float k_val = to_float(K_ptr[kv * k_strideM + d]);
            dot += q_val * k_val;
        }
        dot *= scale;
        row_max = fmaxf(row_max, dot);
    }
    
    // Warp reduce max
    row_max = warp_max(row_max);
    
    // Cross-warp reduction
    if (blockDim.x > WARP_SIZE) {
        int warp_id = tid / WARP_SIZE;
        int num_warps = blockDim.x / WARP_SIZE;
        
        if (lane == 0) smem[warp_id] = row_max;
        __syncthreads();
        
        if (warp_id == 0) {
            row_max = (lane < num_warps) ? smem[lane] : -INFINITY;
            row_max = warp_max(row_max);
            if (lane == 0) smem[0] = row_max;
        }
        __syncthreads();
        row_max = smem[0];
        __syncthreads();
    }
    
    // Step 2: Compute softmax and weighted sum with V
    float row_sum = 0.0f;
    
    // Accumulators for output (each thread handles subset of dimensions)
    float acc[8] = {0};
    int acc_per_thread = (head_dim_v + blockDim.x - 1) / blockDim.x;
    int d_start = tid * acc_per_thread;
    int d_end = min(d_start + acc_per_thread, head_dim_v);
    
    // Actually, use warp-strided access for coalescing
    // Each thread in warp handles different dimension
    
    for (int kv = 0; kv < seq_kv; kv++) {
        if (causal && kv > q_row) continue;
        
        // Compute attention score
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            float q_val = to_float(Q_ptr[q_row * q_strideM + d]);
            float k_val = to_float(K_ptr[kv * k_strideM + d]);
            dot += q_val * k_val;
        }
        dot *= scale;
        
        float exp_val = expf(dot - row_max);
        row_sum += exp_val;
        
        // Accumulate V weighted by exp_val
        // Each warp handles all dimensions, lanes stride across them
        for (int d = lane; d < head_dim_v; d += WARP_SIZE) {
            float v_val = to_float(V_ptr[kv * v_strideM + d]);
            acc[d / WARP_SIZE] += exp_val * v_val;
        }
    }
    
    // Reduce row_sum
    row_sum = warp_sum(row_sum);
    if (blockDim.x > WARP_SIZE) {
        int warp_id = tid / WARP_SIZE;
        int num_warps = blockDim.x / WARP_SIZE;
        
        if (lane == 0) smem[warp_id] = row_sum;
        __syncthreads();
        
        if (warp_id == 0) {
            row_sum = (lane < num_warps) ? smem[lane] : 0.0f;
            row_sum = warp_sum(row_sum);
            if (lane == 0) smem[0] = row_sum;
        }
        __syncthreads();
        row_sum = smem[0];
    }
    
    // Write output
    float inv_sum = 1.0f / row_sum;
    for (int d = lane; d < head_dim_v; d += WARP_SIZE) {
        float val = acc[d / WARP_SIZE] * inv_sum;
        O_ptr[q_row * o_strideM + d] = to_half(val);
    }
}

// More efficient kernel with better memory access patterns
__global__ void fmha_kernel_v2(
    int batch_size, int seq_q, int seq_kv,
    int num_heads, int head_dim, int head_dim_v,
    __half const *Q, __half const *K, __half const *V, __half *O,
    int q_strideB, int q_strideM, int q_strideH,
    int k_strideB, int k_strideM, int k_strideH,
    int v_strideB, int v_strideM, int v_strideH,
    int o_strideM,
    bool causal) {
    
    // Block: (batch*head, query_row)
    // Thread: handles multiple KV positions and V dimensions
    
    int bh_idx = blockIdx.x;
    int batch = bh_idx / num_heads;
    int head = bh_idx % num_heads;
    int q_row = blockIdx.y;
    
    if (q_row >= seq_q) return;
    
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    
    // Pointers
    const __half* q_row_ptr = Q + batch * q_strideB + head * q_strideH + q_row * q_strideM;
    const __half* K_ptr = K + batch * k_strideB + head * k_strideH;
    const __half* V_ptr = V + batch * v_strideB + head * v_strideH;
    __half* o_row_ptr = O + batch * (seq_q * o_strideM) + q_row * o_strideM + head * head_dim_v;
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Shared memory: reduction workspace + Q row cache
    extern __shared__ char smem_raw[];
    float* smem = (float*)smem_raw;
    
    // Cache Q row in registers/shared for reuse
    float q_cache[64];  // Max head_dim
    #pragma unroll 8
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_cache[d] = to_float(q_row_ptr[d]);
    }
    
    // Ensure all threads have loaded Q
    __syncthreads();
    
    // Step 1: Find max attention score
    float local_max = -INFINITY;
    
    // Each thread processes multiple KV positions
    for (int kv = tid; kv < seq_kv; kv += blockDim.x) {
        if (causal && kv > q_row) continue;
        
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            float k_val = to_float(K_ptr[kv * k_strideM + d]);
            dot += q_cache[d] * k_val;
        }
        dot *= scale;
        local_max = fmaxf(local_max, dot);
    }
    
    // Reduce max
    float row_max = local_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
    }
    
    if (blockDim.x > WARP_SIZE) {
        int num_warps = blockDim.x / WARP_SIZE;
        if (lane == 0) smem[warp] = row_max;
        __syncthreads();
        
        if (warp == 0) {
            row_max = (lane < num_warps) ? smem[lane] : -INFINITY;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
            }
            if (lane == 0) smem[0] = row_max;
        }
        __syncthreads();
        row_max = smem[0];
        __syncthreads();
    }
    
    // Step 2: Compute softmax and aggregate V
    float row_sum = 0.0f;
    float out_acc[64] = {0};  // Accumulators for output
    
    for (int kv = tid; kv < seq_kv; kv += blockDim.x) {
        if (causal && kv > q_row) continue;
        
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            float k_val = to_float(K_ptr[kv * k_strideM + d]);
            dot += q_cache[d] * k_val;
        }
        dot *= scale;
        
        float exp_val = expf(dot - row_max);
        row_sum += exp_val;
        
        // Accumulate weighted V
        #pragma unroll 4
        for (int d = 0; d < head_dim_v; d++) {
            float v_val = to_float(V_ptr[kv * v_strideM + d]);
            out_acc[d] += exp_val * v_val;
        }
    }
    
    // Reduce sum
    float total_sum = row_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);
    }
    
    if (blockDim.x > WARP_SIZE) {
        int num_warps = blockDim.x / WARP_SIZE;
        if (lane == 0) smem[warp] = total_sum;
        __syncthreads();
        
        if (warp == 0) {
            total_sum = (lane < num_warps) ? smem[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                total_sum += __shfl_down_sync(0xffffffff, total_sum, offset);
            }
            if (lane == 0) smem[0] = total_sum;
        }
        __syncthreads();
        total_sum = smem[0];
        __syncthreads();
    }
    
    // Step 3: Write output (normalize and store)
    float inv_sum = 1.0f / total_sum;
    
    // Reduce accumulators across threads and write
    // Each dimension needs to be summed across all threads
    
    for (int d = 0; d < head_dim_v; d++) {
        float val = out_acc[d];
        // Sum across all threads
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        
        if (blockDim.x > WARP_SIZE) {
            int num_warps = blockDim.x / WARP_SIZE;
            if (lane == 0) smem[warp + d * num_warps] = val;
            __syncthreads();
            
            if (warp == 0) {
                val = (lane < num_warps) ? smem[lane + d * num_warps] : 0.0f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xffffffff, val, offset);
                }
                if (lane == 0) smem[d] = val;
            }
            __syncthreads();
            val = smem[d];
        }
        
        if (tid == 0) {
            o_row_ptr[d] = to_half(val * inv_sum);
        }
    }
}

// Final optimized version with proper thread cooperation
__global__ void fmha_kernel_final(
    int batch_size, int seq_q, int seq_kv,
    int num_heads, int head_dim, int head_dim_v,
    __half const *Q, __half const *K, __half const *V, __half *O,
    int q_strideB, int q_strideM, int q_strideH,
    int k_strideB, int k_strideM, int k_strideH,
    int v_strideB, int v_strideM, int v_strideH,
    int o_strideM,
    bool causal) {
    
    // Configuration: 128 threads per block
    // Each block handles 1 query row
    // Threads cooperate on both KV loop and V dimensions
    
    int bh_idx = blockIdx.x;
    int batch = bh_idx / num_heads;
    int head = bh_idx % num_heads;
    int q_row = blockIdx.y;
    
    if (q_row >= seq_q) return;
    
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const int NUM_WARPS = 4;  // 128 threads
    
    // Pointers
    const __half* q_row_ptr = Q + batch * q_strideB + head * q_strideH + q_row * q_strideM;
    const __half* K_ptr = K + batch * k_strideB + head * k_strideH;
    const __half* V_ptr = V + batch * v_strideB + head * v_strideH;
    __half* o_row_ptr = O + batch * (seq_q * o_strideM) + q_row * o_strideM + head * head_dim_v;
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Shared memory layout:
    // - smem[0..NUM_WARPS-1]: for max reduction
    // - smem[NUM_WARPS..2*NUM_WARPS-1]: for sum reduction  
    // - smem[2*NUM_WARPS..]: for output reduction (head_dim_v * NUM_WARPS)
    extern __shared__ float smem[];
    float* max_smem = smem;
    float* sum_smem = smem + NUM_WARPS;
    float* out_smem = smem + 2 * NUM_WARPS;
    
    // Cache Q row in registers (cooperative load)
    float q_cache[64];
    #pragma unroll 4
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_cache[d] = to_float(q_row_ptr[d]);
    }
    __syncthreads();
    
    // Step 1: Compute max attention score
    float local_max = -INFINITY;
    
    // Strided loop over KV
    for (int kv = tid; kv < seq_kv; kv += blockDim.x) {
        if (causal && kv > q_row) continue;
        
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            dot += q_cache[d] * to_float(K_ptr[kv * k_strideM + d]);
        }
        dot *= scale;
        local_max = fmaxf(local_max, dot);
    }
    
    // Warp reduce
    float warp_max_val = local_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max_val = fmaxf(warp_max_val, __shfl_down_sync(0xffffffff, warp_max_val, offset));
    }
    
    // Store warp max
    if (lane == 0) max_smem[warp] = warp_max_val;
    __syncthreads();
    
    // Final reduce
    if (warp == 0) {
        float val = (lane < NUM_WARPS) ? max_smem[lane] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) max_smem[0] = val;
    }
    __syncthreads();
    float row_max = max_smem[0];
    
    // Step 2: Compute softmax and weighted sum
    float local_sum = 0.0f;
    
    // Each warp handles a subset of V dimensions
    // Warp 0: dims 0-15, Warp 1: dims 16-31, etc. (assuming head_dim_v=64)
    int v_per_warp = (head_dim_v + NUM_WARPS - 1) / NUM_WARPS;
    int v_start = warp * v_per_warp;
    int v_end = min(v_start + v_per_warp, head_dim_v);
    
    // Local accumulators for this warp's V dimensions
    float acc[16] = {0};  // Max 16 per warp for head_dim_v=64
    
    for (int kv = tid; kv < seq_kv; kv += blockDim.x) {
        if (causal && kv > q_row) continue;
        
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            dot += q_cache[d] * to_float(K_ptr[kv * k_strideM + d]);
        }
        dot *= scale;
        
        float exp_val = expf(dot - row_max);
        local_sum += exp_val;
        
        // Load V and accumulate
        #pragma unroll 4
        for (int vi = 0; vi < v_per_warp && v_start + vi < head_dim_v; vi++) {
            int d = v_start + vi;
            acc[vi] += exp_val * to_float(V_ptr[kv * v_strideM + d]);
        }
    }
    
    // Reduce sum across warps
    float warp_sum_val = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum_val += __shfl_down_sync(0xffffffff, warp_sum_val, offset);
    }
    if (lane == 0) sum_smem[warp] = warp_sum_val;
    __syncthreads();
    
    if (warp == 0) {
        float val = (lane < NUM_WARPS) ? sum_smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) sum_smem[0] = val;
    }
    __syncthreads();
    float row_sum = sum_smem[0];
    float inv_sum = 1.0f / row_sum;
    
    // Step 3: Write output
    // Each warp writes its V dimensions
    // Need to store accumulators to shared memory first for proper indexing
    
    // Store accumulators to shared memory
    for (int vi = 0; vi < v_per_warp && v_start + vi < head_dim_v; vi++) {
        if (lane == 0) {
            out_smem[warp + vi * NUM_WARPS] = acc[vi];
        }
    }
    __syncthreads();
    
    // Now write to global memory (coalesced)
    // Each thread writes one element
    for (int d = tid; d < head_dim_v; d += blockDim.x) {
        int warp_id = d / v_per_warp;
        int local_idx = d % v_per_warp;
        float val = out_smem[warp_id + local_idx * NUM_WARPS];
        o_row_ptr[d] = to_half(val * inv_sum);
    }
}

// Simpler but correct version
__global__ void fmha_kernel_simple(
    int batch_size, int seq_q, int seq_kv,
    int num_heads, int head_dim, int head_dim_v,
    __half const *Q, __half const *K, __half const *V, __half *O,
    int q_strideB, int q_strideM, int q_strideH,
    int k_strideB, int k_strideM, int k_strideH,
    int v_strideB, int v_strideM, int v_strideH,
    int o_strideM,
    bool causal) {
    
    int bh_idx = blockIdx.x;
    int batch = bh_idx / num_heads;
    int head = bh_idx % num_heads;
    int q_row = blockIdx.y;
    
    if (q_row >= seq_q) return;
    
    int tid = threadIdx.x;
    
    // Pointers
    const __half* q_row_ptr = Q + batch * q_strideB + head * q_strideH + q_row * q_strideM;
    const __half* K_ptr = K + batch * k_strideB + head * k_strideH;
    const __half* V_ptr = V + batch * v_strideB + head * v_strideH;
    __half* o_row_ptr = O + batch * (seq_q * o_strideM) + q_row * o_strideM + head * head_dim_v;
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    extern __shared__ float smem[];
    
    // Step 1: Load Q to shared memory (cooperative)
    // Actually, use registers - each thread loads full Q row? No, too big
    
    // Instead: each thread computes partial dot products
    // and we use shared memory for reduction
    
    // First, find max score
    float thread_max = -INFINITY;
    
    for (int kv = tid; kv < seq_kv; kv += blockDim.x) {
        if (causal && kv > q_row) continue;
        
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += to_float(q_row_ptr[d]) * to_float(K_ptr[kv * k_strideM + d]);
        }
        dot *= scale;
        thread_max = fmaxf(thread_max, dot);
    }
    
    // Block reduction for max
    // Use shared memory
    smem[tid] = thread_max;
    __syncthreads();
    
    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    float row_max = smem[0];
    
    // Step 2: Compute softmax and sum
    float thread_sum = 0.0f;
    float thread_out[64] = {0};  // Accumulators
    
    for (int kv = tid; kv < seq_kv; kv += blockDim.x) {
        if (causal && kv > q_row) continue;
        
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += to_float(q_row_ptr[d]) * to_float(K_ptr[kv * k_strideM + d]);
        }
        dot *= scale;
        
        float exp_val = expf(dot - row_max);
        thread_sum += exp_val;
        
        for (int d = 0; d < head_dim_v; d++) {
            thread_out[d] += exp_val * to_float(V_ptr[kv * v_strideM + d]);
        }
    }
    
    // Sum reduction
    smem[tid] = thread_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = smem[0];
    float inv_sum = 1.0f / row_sum;
    
    // Output reduction - need separate shared memory
    // Use smem+blockDim.x onwards
    float* out_smem = smem + blockDim.x;
    
    for (int d = 0; d < head_dim_v; d++) {
        // Each thread has partial sum in thread_out[d]
        smem[tid] = thread_out[d];
        __syncthreads();
        
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                smem[tid] += smem[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            o_row_ptr[d] = to_half(smem[0] * inv_sum);
        }
    }
}

// Optimized version with better memory access
__global__ void fmha_kernel_opt(
    int batch_size, int seq_q, int seq_kv,
    int num_heads, int head_dim, int head_dim_v,
    __half const *Q, __half const *K, __half const *V, __half *O,
    int q_strideB, int q_strideM, int q_strideH,
    int k_strideB, int k_strideM, int k_strideH,
    int v_strideB, int v_strideM, int v_strideH,
    int o_strideM,
    bool causal) {
    
    int bh_idx = blockIdx.x;
    int batch = bh_idx / num_heads;
    int head = bh_idx % num_heads;
    int q_row = blockIdx.y;
    
    if (q_row >= seq_q) return;
    
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    
    const int NUM_WARPS = blockDim.x >> 5;
    
    // Pointers
    const __half* q_row_ptr = Q + batch * q_strideB + head * q_strideH + q_row * q_strideM;
    const __half* K_ptr = K + batch * k_strideB + head * k_strideH;
    const __half* V_ptr = V + batch * v_strideB + head * v_strideH;
    __half* o_row_ptr = O + batch * (seq_q * o_strideM) + q_row * o_strideM + head * head_dim_v;
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    extern __shared__ float smem[];
    float* warp_smem = smem;  // size NUM_WARPS
    
    // Cache Q in registers (cooperative load)
    __shared__ float q_shared[128];  // Max head_dim
    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = to_float(q_row_ptr[d]);
    }
    __syncthreads();
    
    // Step 1: Find max
    float local_max = -INFINITY;
    for (int kv = tid; kv < seq_kv; kv += blockDim.x) {
        if (causal && kv > q_row) continue;
        
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            dot += q_shared[d] * to_float(K_ptr[kv * k_strideM + d]);
        }
        dot *= scale;
        local_max = fmaxf(local_max, dot);
    }
    
    // Warp reduce
    float warp_max = local_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_max = fmaxf(warp_max, __shfl_down_sync(0xffffffff, warp_max, offset));
    }
    if (lane == 0) warp_smem[warp] = warp_max;
    __syncthreads();
    
    // Block reduce
    if (warp == 0) {
        float val = (lane < NUM_WARPS) ? warp_smem[lane] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if (lane == 0) warp_smem[0] = val;
    }
    __syncthreads();
    float row_max = warp_smem[0];
    
    // Step 2: Compute softmax and aggregate
    float local_sum = 0.0f;
    
    // Each warp handles different V dimensions
    int v_per_warp = (head_dim_v + NUM_WARPS - 1) / NUM_WARPS;
    int v_start = warp * v_per_warp;
    int v_end = min(v_start + v_per_warp, head_dim_v);
    
    float acc[16] = {0};
    
    for (int kv = tid; kv < seq_kv; kv += blockDim.x) {
        if (causal && kv > q_row) continue;
        
        float dot = 0.0f;
        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            dot += q_shared[d] * to_float(K_ptr[kv * k_strideM + d]);
        }
        dot *= scale;
        
        float exp_val = expf(dot - row_max);
        local_sum += exp_val;
        
        // Accumulate V
        for (int d = v_start; d < v_end; d++) {
            acc[d - v_start] += exp_val * to_float(V_ptr[kv * v_strideM + d]);
        }
    }
    
    // Reduce sum
    float warp_sum = local_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }
    if (lane == 0) warp_smem[warp] = warp_sum;
    __syncthreads();
    
    if (warp == 0) {
        float val = (lane < NUM_WARPS) ? warp_smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) warp_smem[0] = val;
    }
    __syncthreads();
    float row_sum = warp_smem[0];
    float inv_sum = 1.0f / row_sum;
    
    // Write output
    // Store accumulators to shared, then write coalesced
    float* out_smem = smem + NUM_WARPS;
    for (int i = 0; i < v_per_warp && v_start + i < head_dim_v; i++) {
        if (lane == 0) out_smem[warp + i * NUM_WARPS] = acc[i];
    }
    __syncthreads();
    
    for (int d = tid; d < head_dim_v; d += blockDim.x) {
        int w = d / v_per_warp;
        int idx = d % v_per_warp;
        float val = out_smem[w + idx * NUM_WARPS];
        o_row_ptr[d] = to_half(val * inv_sum);
    }
}

// Main host function
cudaError_t fmha_solution(
    int batch_size, int seq_q, int seq_kv,
    int num_heads, int head_dim, int head_dim_v,
    __half const *Q, __half const *K, __half const *V, __half *O,
    int q_strideB, int q_strideM, int q_strideH,
    int k_strideB, int k_strideM, int k_strideH,
    int v_strideB, int v_strideM, int v_strideH,
    int o_strideM,
    bool causal) {
    
    // Grid: (batch * num_heads, seq_q)
    // Block: 128 threads
    
    dim3 grid(batch_size * num_heads, seq_q);
    dim3 block(128);
    
    // Shared memory: warp results + output staging
    // NUM_WARPS=4, so 4 floats for max, 4 for sum, 4*64 for output
    size_t smem_size = (4 + 4 + 4 * head_dim_v) * sizeof(float);
    
    // Ensure we have enough shared memory
    if (smem_size > 48 * 1024) {
        // Fall back to simpler version with less shared memory
        smem_size = (128 + 128) * sizeof(float);  // Just for reductions
        
        fmha_kernel_simple<<<grid, block, smem_size>>>(
            batch_size, seq_q, seq_kv,
            num_heads, head_dim, head_dim_v,
            Q, K, V, O,
            q_strideB, q_strideM, q_strideH,
            k_strideB, k_strideM, k_strideH,
            v_strideB, v_strideM, v_strideH,
            o_strideM,
            causal);
    } else {
        fmha_kernel_opt<<<grid, block, smem_size>>>(
            batch_size, seq_q, seq_kv,
            num_heads, head_dim, head_dim_v,
            Q, K, V, O,
            q_strideB, q_strideM, q_strideH,
            k_strideB, k_strideM, k_strideH,
            v_strideB, v_strideM, v_strideH,
            o_strideM,
            causal);
    }
    
    return cudaGetLastError();
}
