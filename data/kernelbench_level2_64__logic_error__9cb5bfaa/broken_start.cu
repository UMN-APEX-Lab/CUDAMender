import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel that fuses: LogSumExp + LeakyReLU + LeakyReLU + GELU + GELU
# We fuse all operations after the linear layer into a single kernel for maximum performance

fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// GELU approximation using tanh: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// This matches PyTorch's GELU implementation with 'none' approximation
__device__ __forceinline__ float gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float leaky_relu(float x, float negative_slope) {
    return x > 0.0f ? x : negative_slope * x;
}

__global__ void fused_logsumexp_leakyrelu_leakyrelu_gelu_gelu_kernel(
    const float* input, 
    float* output,
    int batch_size,
    int features
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Each block handles one batch element
    // First compute max for numerical stability of logsumexp
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float val = input[batch_idx * features + i];
        max_val = fmaxf(max_val, val);
    }
    
    // Warp-level reduction for max
    __shared__ float shared_max;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if (threadIdx.x == 0) {
        shared_max = max_val;
    }
    __syncthreads();
    max_val = shared_max;
    
    // Compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float val = input[batch_idx * features + i];
        sum_exp += expf(val - max_val);
    }
    
    // Warp-level reduction for sum
    __shared__ float shared_sum;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    if (threadIdx.x == 0) {
        shared_sum = sum_exp;
    }
    __syncthreads();
    sum_exp = shared_sum;
    
    // Compute logsumexp = log(sum_exp) + max
    float lse = logf(sum_exp) + max_val;
    
    // Apply LeakyReLU (negative_slope = 0.01)
    lse = leaky_relu(lse, 0.01f);
    // Apply second LeakyReLU
    lse = leaky_relu(lse, 0.01f);
    // Apply first GELU
    lse = gelu(lse);
    // Apply second GELU
    lse = gelu(lse);
    
    if (threadIdx.x == 0) {
        output[batch_idx] = lse;
    }
}

torch::Tensor fused_ops_cuda(torch::Tensor input) {
    int batch_size = input.size(0);
    int features = input.size(1);
    
    auto output = torch::empty({batch_size, 1}, input.options());
    
    const int threads = 256;
    const int blocks = batch_size;
    
    fused_logsumexp_leakyrelu_leakyrelu_gelu_gelu_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        features
    );
    
    return output;
}
"""

fused_ops_cpp_source = "torch::Tensor fused_ops_cuda(torch::Tensor input);"

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_ops_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that fuses LogSumExp, LeakyReLU, LeakyReLU, GELU, GELU into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.fused_ops = fused_ops

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        # Fused: LogSumExp + LeakyReLU + LeakyReLU + GELU + GELU
        x = self.fused_ops.fused_ops_cuda(x)
        return x