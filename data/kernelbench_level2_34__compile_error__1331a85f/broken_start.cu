import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# We fuse LayerNorm, GELU, and Scaling into a single CUDA kernel.
# ConvTranspose3d is highly optimized in cuDNN, so we keep the PyTorch implementation.
# LayerNorm is applied over the channel dimension (out_channels).
# Input to the fused kernel: (batch, channels, D, H, W)
# LayerNorm expects normalization over the last dimension, but here channels is the 2nd dim.
# We treat the spatial dimensions as part of the batch for the LN kernel.

fused_ln_gelu_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_ln_gelu_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int channels,
    int spatial_size,
    float eps,
    float scaling_factor) 
{
    // Each block handles one 'pixel' across all channels
    // input shape: [batch * spatial_size, channels]
    int pixel_idx = blockIdx.x; 
    int tid = threadIdx.x;

    extern __shared__ float shared_mem[];
    float* s_mean = shared_mem;
    float* s_var = &shared_mem[1];

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int i = tid; i < channels; i += blockDim.x) {
        float val = input[pixel_idx * channels + i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Warp reduction for sum and sq_sum
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sq_sum += __shfl_down_sync(0xffffffff, local_sq_sum, offset);
    }

    if (tid == 0) {
        float mean = local_sum / channels;
        float var = (local_sq_sum / channels) - (mean * mean);
        *s_mean = mean;
        *s_var = var;
    }
    __syncthreads();

    float mean = *s_mean;
    float inv_std = 1.0f / sqrtf(*s_var + eps);

    for (int i = tid; i < channels; i += blockDim.x) {
        int idx = pixel_idx * channels + i;
        float val = (input[idx] - mean) * inv_std;
        
        // Apply LayerNorm weight and bias
        val = val * weight[i] + bias[i];
        
        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x = val;
        float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        
        // Scaling
        output[idx] = gelu * scaling_factor;
    }
}

torch::Tensor fused_ln_gelu_scale_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps, float scaling_factor) {
    auto input_shape = x.sizes(); // [B, C, D, H, W]
    int B = input_shape[0];
    int C = input_shape[1];
    int D = input_shape[2];
    int H = input_shape[3];
    int W = input_shape[4];
    int spatial_size = D * H * W;
    
    // Permute to [B, D, H, W, C] to make channels contiguous for LN
    auto x_permuted = x.permute({0, 2, 3, 4, 1}).contiguous();
    auto output = torch::empty_like(x_permuted);

    int total_pixels = B * spatial_size;
    const int block_size = 256;
    
    fused_ln_gelu_scale_kernel<<<total_pixels, block_size, 2 * sizeof(float)>>>(
        x_permuted.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        C,
        spatial_size,
        eps,
        scaling_factor
    );

    // Permute back to [B, C, D, H, W]
    return output.permute({0, 4, 1, 2, 3}).contiguous();
}
"""

fused_ln_gelu_scale_cpp_source = "torch::Tensor fused_ln_gelu_scale_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, float eps, float scaling_factor);"

fused_op = load_inline(
    name="fused_ln_gelu_scale",
    cpp_sources=fused_ln_gelu_scale_cpp_source,
    cuda_sources=fused_ln_gelu_scale_source,
    functions=["fused_ln_gelu_scale_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        # The fused kernel expects (B, C, D, H, W) and performs LN over C
        x = fused_op.fused_ln_gelu_scale_cuda(
            x, 
            self.layer_norm.weight, 
            self.layer_norm.bias, 
            self.eps, 
            self.scaling_factor
        )
        return x