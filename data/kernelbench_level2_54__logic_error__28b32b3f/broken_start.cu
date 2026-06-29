import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused conv + multiply + leaky_relu + gelu
fused_conv_mult_leaky_gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float gelu_approx(float x) {
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // Using the faster approximation: 0.5 * x * (1 + tanh(0.7978845608f * (x + 0.044715f * x * x * x)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__device__ __forceinline__ float leaky_relu(float x, float negative_slope) {
    return x > 0 ? x : negative_slope * x;
}

__global__ void fused_conv_mult_leaky_gelu_kernel(
    const float* conv_out, 
    const float* multiplier,
    float* out,
    int batch_size,
    int channels,
    int height,
    int width,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height * width;
    
    if (idx >= total_elements) return;
    
    // Compute indices
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    int b = idx / (width * height * channels);
    
    // Load conv output
    float val = conv_out[idx];
    
    // Multiply by learnable scalar (broadcast over batch, height, width)
    val = val * multiplier[c];
    
    // Apply LeakyReLU
    val = leaky_relu(val, negative_slope);
    
    // Apply GELU
    val = gelu_approx(val);
    
    out[idx] = val;
}

torch::Tensor fused_conv_mult_leaky_gelu_cuda(
    torch::Tensor conv_out,
    torch::Tensor multiplier,
    float negative_slope
) {
    int batch_size = conv_out.size(0);
    int channels = conv_out.size(1);
    int height = conv_out.size(2);
    int width = conv_out.size(3);
    
    auto out = torch::empty_like(conv_out);
    
    int total_elements = batch_size * channels * height * width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    fused_conv_mult_leaky_gelu_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        multiplier.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        negative_slope
    );
    
    return out;
}
"""

fused_conv_mult_leaky_gelu_cpp_source = (
    "torch::Tensor fused_conv_mult_leaky_gelu_cuda(torch::Tensor conv_out, torch::Tensor multiplier, float negative_slope);"
)

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_conv_mult_leaky_gelu",
    cpp_sources=fused_conv_mult_leaky_gelu_cpp_source,
    cuda_sources=fused_conv_mult_leaky_gelu_source,
    functions=["fused_conv_mult_leaky_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a convolution, multiplies by a learnable scalar, 
    applies LeakyReLU, and then GELU using a fused custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape)) 
        self.negative_slope = 0.01  # Default LeakyReLU negative slope
        self.fused_ops = fused_ops

    def forward(self, x):
        x = self.conv(x)
        x = self.fused_ops.fused_conv_mult_leaky_gelu_cuda(x, self.multiplier, self.negative_slope)
        return x


def get_inputs():
    batch_size = 64
    in_channels = 64
    height, width = 256, 256
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    in_channels = 64
    out_channels = 64
    kernel_size = 3
    multiplier_shape = (64, 1, 1)
    return [in_channels, out_channels, kernel_size, multiplier_shape]