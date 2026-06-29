import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel that fuses: conv_transpose output * multiplier + two global average poolings
# The key insight: after first global avg pool, we get [B, C, 1, 1], second pool is redundant
# We can also fuse the multiplication with the pooling

fused_conv_transpose_gpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

// Fused kernel: conv_transpose output * multiplier, then global avg pool (both dims at once)
__global__ void fused_conv_transpose_gpool_kernel(
    const float* conv_out,  // [B, C, H, W]
    float* out,             // [B, C, 1, 1]
    int B, int C, int H, int W,
    float multiplier
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    
    if (idx >= total) return;
    
    int b = idx / C;
    int c = idx % C;
    
    float sum = 0.0f;
    int base_idx = b * C * H * W + c * H * W;
    
    #pragma unroll 4
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            sum += conv_out[base_idx + h * W + w] * multiplier;
        }
    }
    
    sum /= (H * W);  // global average pooling
    
    // Second pooling is redundant since we already have [B, C, 1, 1]
    // But we need to output [B, C, 1, 1] to match original semantics
    out[idx] = sum;
}

// Wrapper that handles the conv_transpose call and fused post-processing
torch::Tensor fused_conv_transpose_gpool_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int in_channels,
    int out_channels,
    float multiplier
) {
    // First compute conv_transpose using cuDNN (via PyTorch)
    // We need to manually compute the output size
    int B = x.size(0);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    
    // ConvTranspose2d output size formula
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_h + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_w + output_padding;
    
    // Use PyTorch's conv_transpose2d
    auto conv_out = torch::nn::functional::conv_transpose2d(
        x, weight, bias,
        torch::nn::functional::ConvTranspose2dFuncOptions()
            .stride(stride)
            .padding(padding)
            .output_padding(output_padding)
            .groups(groups)
    );
    
    // Now apply fused kernel: multiply by multiplier and global avg pool
    auto out = torch::empty({B, out_channels, 1, 1}, x.options());
    
    const int block_size = 256;
    const int num_blocks = (B * out_channels + block_size - 1) / block_size;
    
    fused_conv_transpose_gpool_kernel<<<num_blocks, block_size>>>(
        conv_out.data_ptr<float>(),
        out.data_ptr<float>(),
        B, out_channels, H_out, W_out,
        multiplier
    );
    
    return out;
}
"""

fused_conv_transpose_gpool_cpp_source = """
torch::Tensor fused_conv_transpose_gpool_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups,
    int in_channels,
    int out_channels,
    float multiplier
);
"""

# Compile the inline CUDA code
fused_conv_transpose_gpool = load_inline(
    name="fused_conv_transpose_gpool",
    cpp_sources=fused_conv_transpose_gpool_cpp_source,
    cuda_sources=fused_conv_transpose_gpool_source,
    functions=["fused_conv_transpose_gpool_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.multiplier = multiplier
        
        # Initialize conv_transpose weights manually
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))
        
        # Initialize using same init as ConvTranspose2d
        nn.init.kaiming_uniform_(self.weight, a=std::sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / std::sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_conv_transpose_gpool

    def forward(self, x):
        return self.fused_op.fused_conv_transpose_gpool_cuda(
            x, self.weight, self.bias,
            self.stride, self.padding, self.output_padding,
            1,  # groups
            self.in_channels,
            self.out_channels,
            self.multiplier
        )