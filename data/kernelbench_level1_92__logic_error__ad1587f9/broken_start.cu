import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for exclusive cumulative sum
exclusive_cumsum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel for exclusive scan using parallel prefix sum algorithm
// Each block processes a portion of the array using shared memory
__global__ void exclusive_scan_kernel(const float* input, float* output, int n) {
    extern __shared__ float temp[];
    
    int tid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory (with shift for exclusive scan)
    // For exclusive scan, we shift everything by 1, putting 0 at position 0
    int ai = tid;
    int bi = tid + blockDim.x;
    
    // First half of shared memory: load with shift
    temp[2*tid] = (ai > 0 && ai <= n) ? input[ai-1] : 0.0f;
    temp[2*tid + 1] = (bi > 0 && bi <= n) ? input[bi-1] : 0.0f;
    
    __syncthreads();
    
    // Up-sweep phase (reduce)
    for (int d = blockDim.x; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    
    // Clear the last element (exclusive scan)
    if (tid == 0) {
        temp[2 * blockDim.x - 1] = 0;
    }
    
    // Down-sweep phase
    for (int d = 1; d < 2 * blockDim.x; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    
    __syncthreads();
    
    // Write results
    output[2*tid] = temp[2*tid];
    output[2*tid + 1] = temp[2*tid + 1];
}

// Simple exclusive scan kernel for when n is small
__global__ void exclusive_scan_simple_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < idx; i++) {
            sum += input[i];
        }
        output[idx] = sum;
    }
}

// Optimized kernel for large arrays using sequential scan per thread
__global__ void exclusive_scan_sequential_kernel(const float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements
    for (int idx = tid; idx < n; idx += num_threads) {
        float sum = 0.0f;
        for (int i = 0; i < idx; i++) {
            sum += input[i];
        }
        output[idx] = sum;
    }
}

// Efficient parallel exclusive scan using warp shuffle
__global__ void exclusive_scan_warp_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    
    if (idx >= n) return;
    
    float val = (idx > 0) ? input[idx - 1] : 0.0f;
    
    // Warp-level inclusive scan
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float y = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += y;
    }
    
    // Get warp prefix sums
    __shared__ float warpSums[32];
    if (lane == 31) warpSums[warp] = val;
    __syncthreads();
    
    // Scan warp sums
    if (warp == 0 && lane < blockDim.x / 32) {
        float t = warpSums[lane];
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float y = __shfl_up_sync(0xffffffff, t, offset);
            if (lane >= offset) t += y;
        }
        warpSums[lane] = t;
    }
    __syncthreads();
    
    // Add warp prefix to each thread
    float warpPrefix = (warp > 0) ? warpSums[warp - 1] : 0.0f;
    val += warpPrefix;
    
    // Convert inclusive to exclusive
    float exclusive = val - ((idx > 0) ? input[idx - 1] : 0.0f);
    output[idx] = exclusive;
}

// Multi-block exclusive scan with auxiliary array for block sums
__global__ void exclusive_scan_block_kernel(const float* input, float* output, float* blockSums, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int blockSize = blockDim.x;
    int globalIdx = bid * blockSize + tid;
    
    // Load data
    float val = 0.0f;
    if (globalIdx < n) {
        val = input[globalIdx];
    }
    
    // Inclusive scan within warp
    int lane = tid % 32;
    int warpId = tid / 32;
    
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        float y = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) val += y;
    }
    
    // Store warp sums
    if (lane == 31) {
        sdata[warpId] = val;
    }
    __syncthreads();
    
    // Scan warp sums
    if (warpId == 0) {
        float t = (tid < blockSize / 32) ? sdata[tid] : 0.0f;
        #pragma unroll
        for (int offset = 1; offset < 32; offset <<= 1) {
            float y = __shfl_up_sync(0xffffffff, t, offset);
            if (lane >= offset) t += y;
        }
        if (tid < blockSize / 32) {
            sdata[tid] = t;
        }
    }
    __syncthreads();
    
    // Add warp prefix
    float warpPrefix = (warpId > 0) ? sdata[warpId - 1] : 0.0f;
    val += warpPrefix;
    
    // Store block sum
    if (tid == blockSize - 1) {
        blockSums[bid] = val;
    }
    
    // Write output (shifted for exclusive scan)
    if (globalIdx < n) {
        output[globalIdx] = val - input[globalIdx];
    }
}

__global__ void add_block_prefix_kernel(float* output, const float* blockSums, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;
    
    if (tid < n && bid > 0) {
        output[tid] += blockSums[bid];
    }
}

// Main exclusive cumsum function for 2D tensor along dimension 1
torch::Tensor exclusive_cumsum_cuda(torch::Tensor x, int dim) {
    auto sizes = x.sizes();
    int batch_size = sizes[0];
    int n = sizes[1];
    
    auto output = torch::empty_like(x);
    
    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    
    // For simplicity and efficiency, process each row independently
    // Using a simple but efficient approach: each thread computes prefix sum for its elements
    
    // Launch kernel for each batch element
    for (int b = 0; b < batch_size; b++) {
        auto row_input = x[b];
        auto row_output = output[b];
        
        // Use simple sequential scan for each row - actually quite efficient for moderate n
        // due to memory coalescing and cache efficiency
        exclusive_scan_sequential_kernel<<<num_blocks, block_size>>>(
            row_input.data_ptr<float>(),
            row_output.data_ptr<float>(),
            n
        );
    }
    
    return output;
}

// Optimized version using single kernel launch with proper indexing
__global__ void exclusive_cumsum_2d_kernel(const float* input, float* output, int batch_size, int n) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int b = blockIdx.y;
    
    if (b >= batch_size) return;
    
    int idx = bid * blockDim.x + tid;
    
    // Simple approach: each thread handles one element, uses shared memory for prefix
    extern __shared__ float s_prefix[];
    
    // Load into shared memory with exclusive shift
    for (int i = tid; i < n; i += blockDim.x) {
        s_prefix[i] = (i > 0) ? input[b * n + i - 1] : 0.0f;
    }
    __syncthreads();
    
    // Parallel prefix sum in shared memory
    for (int offset = 1; offset < n; offset *= 2) {
        __syncthreads();
        for (int i = tid; i < n; i += blockDim.x) {
            if (i >= offset) {
                float temp = s_prefix[i - offset];
                __syncthreads();
                s_prefix[i] += temp;
                __syncthreads();
            }
        }
    }
    __syncthreads();
    
    // Write output
    for (int i = tid; i < n; i += blockDim.x) {
        output[b * n + i] = s_prefix[i];
    }
}

// More efficient: use warp shuffle for better performance
__global__ void exclusive_cumsum_2d_shuffle_kernel(const float* input, float* output, int batch_size, int n) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    
    if (b >= batch_size) return;
    
    const int row_offset = b * n;
    
    // Each thread processes elements with stride
    float running_sum = 0.0f;
    
    for (int i = tid; i < n; i += blockDim.x) {
        // Load input with exclusive shift
        float val = (i > 0) ? input[row_offset + i - 1] : 0.0f;
        
        // Inclusive scan within thread's local elements
        float local_sum = val;
        float prefix = running_sum;
        
        // Store result
        output[row_offset + i] = prefix + local_sum;
        
        // Update running sum for next iteration
        running_sum += val;
    }
    
    // Need to accumulate across threads - use shared memory
    __shared__ float block_sums[256];
    block_sums[tid] = running_sum;
    __syncthreads();
    
    // Prefix sum of block sums
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        __syncthreads();
        float val = 0.0f;
        if (tid >= offset) {
            val = block_sums[tid - offset];
        }
        __syncthreads();
        block_sums[tid] += val;
        __syncthreads();
    }
    
    // Add prefix to output
    float my_prefix = (tid > 0) ? block_sums[tid - 1] : 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        output[row_offset + i] += my_prefix;
    }
}

// Most efficient: single-pass with proper parallel scan
__global__ void exclusive_cumsum_final_kernel(const float* input, float* output, int batch_size, int n) {
    int b = blockIdx.y;
    int tid = threadIdx.x;
    
    if (b >= batch_size) return;
    
    extern __shared__ float sdata[];
    float* temp = sdata;
    
    int row_offset = b * n;
    
    // Load data with exclusive shift (element i gets input[i-1])
    for (int idx = tid; idx < n; idx += blockDim.x) {
        temp[idx] = (idx > 0) ? input[row_offset + idx - 1] : 0.0f;
    }
    __syncthreads();
    
    // Hillis-Steele parallel prefix sum
    for (int d = 1; d < n; d *= 2) {
        __syncthreads();
        for (int idx = tid; idx < n; idx += blockDim.x) {
            if (idx >= d) {
                temp[idx] += temp[idx - d];
            }
        }
    }
    __syncthreads();
    
    // Write output
    for (int idx = tid; idx < n; idx += blockDim.x) {
        output[row_offset + idx] = temp[idx];
    }
}

// Optimized kernel with better occupancy
__global__ void exclusive_cumsum_optimized_kernel(const float* input, float* output, int batch_size, int n) {
    extern __shared__ float shared_mem[];
    
    int b = blockIdx.x;
    int tid = threadIdx.x;
    
    if (b >= batch_size) return;
    
    int row_offset = b * n;
    float* sdata = shared_mem;
    
    // Cooperatively load data
    for (int i = tid; i < n; i += blockDim.x) {
        sdata[i] = (i > 0) ? input[row_offset + i - 1] : 0.0f;
    }
    __syncthreads();
    
    // Parallel prefix sum using Brent-Kung algorithm
    // Up-sweep
    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        for (int i = tid; i < d; i += blockDim.x) {
            int ai = offset * (2 * i + 1) - 1;
            int bi = offset * (2 * i + 2) - 1;
            if (bi < n) {
                sdata[bi] += sdata[ai];
            }
        }
        offset *= 2;
    }
    
    // Clear last element for exclusive scan
    if (tid == 0) {
        sdata[n - 1] = 0;
    }
    __syncthreads();
    
    // Down-sweep
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        for (int i = tid; i < d; i += blockDim.x) {
            int ai = offset * (2 * i + 1) - 1;
            int bi = offset * (2 * i + 2) - 1;
            if (bi < n) {
                float t = sdata[ai];
                sdata[ai] = sdata[bi];
                sdata[bi] += t;
            }
        }
    }
    __syncthreads();
    
    // Write output
    for (int i = tid; i < n; i += blockDim.x) {
        output[row_offset + i] = sdata[i];
    }
}

torch::Tensor exclusive_cumsum_cuda_optimized(torch::Tensor x, int dim) {
    auto sizes = x.sizes();
    int batch_size = sizes[0];
    int n = sizes[1];
    
    auto output = torch::empty_like(x);
    
    // Use 2D grid: x-dimension for batch, but we use 1D grid with batch as x
    const int block_size = 256;
    
    // Dynamically allocate shared memory for the row
    size_t shared_mem_size = n * sizeof(float);
    
    exclusive_cumsum_optimized_kernel<<<batch_size, block_size, shared_mem_size>>>(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        n
    );
    
    return output;
}
"""

exclusive_cumsum_cpp_source = (
    "torch::Tensor exclusive_cumsum_cuda_optimized(torch::Tensor x, int dim);"
)

# Compile the inline CUDA code
exclusive_cumsum = load_inline(
    name="exclusive_cumsum",
    cpp_sources=exclusive_cumsum_cpp_source,
    cuda_sources=exclusive_cumsum_source,
    functions=["exclusive_cumsum_cuda_optimized"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs an exclusive cumulative sum using custom CUDA kernel.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.exclusive_cumsum = exclusive_cumsum

    def forward(self, x):
        # Use custom CUDA kernel for exclusive cumsum along dimension 1
        return self.exclusive_cumsum.exclusive_cumsum_cuda_optimized(x, self.dim)


def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]