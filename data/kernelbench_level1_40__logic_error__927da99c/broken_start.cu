import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Layer Normalization
# This implementation computes mean and variance in a single pass per row (last dimension)
# and then applies the normalization and affine transformation.
layernorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int rows,
    int cols,
    float eps) {
    
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared_mem[];
    float* s_mean = shared_mem;
    float* s_var = &shared_mem[1];

    float sum = 0.0f;
    float sq_sum = 0.0f;

    // Parallel reduction for mean and variance using a simple loop for clarity 
    // in this context, though warp-shuffles would be faster.
    // For cols=256, a simple loop is often sufficient or we can use a block-wide reduction.
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[row * cols + i];
        sum += val;
        sq_sum += val * val;
    }

    // Block-wide reduction using shared memory
    __shared__ float block_sum[32];
    __shared__ float block_sq_sum[32];

    int tid = threadIdx.x;
    block_sum[tid % 32] = 0;
    block_sq_sum[tid % 32] = 0;
    
    // This is a simplified reduction for brevity
    // In a production kernel, we'd use __shfl_down_sync
    float local_sum = sum;
    float local_sq_sum = sq_sum;
    
    // We use a simple atomic-like approach or a structured reduction
    // For a fixed block size of 256, we can do this:
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // This part is tricky in a single kernel without warp primitives
            // Let's use a simpler approach: only thread 0 does the final sum
        }
    }
}
"""

# Since a fully optimized LayerNorm is complex, we implement a robust version 
# using a simpler CUDA approach or leveraging PyTorch's C++ API for the heavy lifting 
# while fusing the affine transform. However, to provide a truly "custom" kernel:

layernorm_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void layernorm_fwd_kernel(
    const float* x, 
    const float* gamma, 
    const float* beta, 
    float* out, 
    int N, int D, float eps) {
    
    int row = blockIdx.x;
    if (row >= N) return;

    float sum = 0.0f;
    float sq_sum = 0.0f;

    for (int i = 0; i < D; ++i) {
        float val = x[row * D + i];
        sum += val;
        sq_sum += val * val;
    }

    float mean = sum / D;
    float var = (sq_sum / D) - (mean * mean);
    float inv_std = 1.0f / sqrtf(var + eps);

    for (int i = 0; i < D; ++i) {
        out[row * D + i] = ((x[row * D + i] - mean) * inv_std) * gamma[i] + beta[i];
    }
}

torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps) {
    auto input_shape = x.sizes();
    int D = input_shape[input_shape.size() - 1];
    int N = x.numel() / D;

    auto out = torch::empty_like(x);

    const int block_size = 1; // One block per row for simplicity in this specific implementation
    const int num_blocks = N;

    layernorm_fwd_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        gamma.data_ptr<float>(), 
        beta.data_ptr<float>(), 
        out.data_ptr<float>(), 
        N, D, eps);

    return out;
}
"""

layernorm_cpp_source = "torch::Tensor layernorm_cuda(torch::Tensor x, torch::Tensor gamma, torch::Tensor beta, float eps);"

layernorm_op = load_inline(
    name="layernorm_op",
    cpp_sources=layernorm_cpp_source,
    cuda_sources=layernorm_cuda_source,
    functions=["layernorm_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super(ModelNew, self).__init__()
        # We maintain the parameters of the original LayerNorm
        self.gamma = nn.Parameter(torch.ones(normalized_shape).cuda())
        self.beta = nn.Parameter(torch.zeros(normalized_shape).cuda())
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten all dimensions except the last one for the CUDA kernel
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]).contiguous()
        
        out_flat = layernorm_op.layernorm_cuda(x_flat, self.gamma, self.beta, self.eps)
        
        return out_flat.view(orig_shape)