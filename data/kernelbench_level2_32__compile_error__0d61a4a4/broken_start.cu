import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel that fuses: Conv2d + scale + min along channels
# Using im2col approach for convolution with fused operations

fused_conv_scale_min_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_conv_scale_min_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int out_height,
    int out_width,
    float scale_factor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = batch_size * out_height * out_width;
    
    if (idx >= total_threads) return;
    
    int b = idx / (out_height * out_width);
    int hw = idx % (out_height * out_width);
    int oh = hw / out_width;
    int ow = hw % out_width;
    
    // Compute min across all output channels for this spatial location
    float min_val = 1e30f;
    
    for (int oc = 0; oc < out_channels; oc++) {
        float sum = 0.0f;
        
        // Convolution computation
        for (int ic = 0; ic < in_channels; ic++) {
            for (int kh = 0; kh < kernel_size; kh++) {
                for (int kw = 0; kw < kernel_size; kw++) {
                    int ih = oh + kh;
                    int iw = ow + kw;
                    int input_idx = ((b * in_channels + ic) * in_height + ih) * in_width + iw;
                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        
        sum += bias[oc];
        sum *= scale_factor;
        
        // Track minimum across channels
        if (sum < min_val) {
            min_val = sum;
        }
    }
    
    int output_idx = ((b * 1 + 0) * out_height + oh) * out_width + ow;
    output[output_idx] = min_val;
}

torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = in_height - kernel_size + 1;
    int out_width = in_width - kernel_size + 1;
    
    auto output = torch::zeros({batch_size, 1, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_height * out_width + block_size - 1) / block_size;
    
    fused_conv_scale_min_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        out_height,
        out_width,
        scale_factor
    );
    
    return output;
}
"""

fused_conv_scale_min_cpp_source = """
torch::Tensor fused_conv_scale_min_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float scale_factor
);
"""

# Compile the inline CUDA code
fused_conv_scale_min = load_inline(
    name="fused_conv_scale_min",
    cpp_sources=fused_conv_scale_min_cpp_source,
    cuda_sources=fused_conv_scale_min_source,
    functions=["fused_conv_scale_min_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        
        # Initialize weights manually to match Conv2d behavior
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
        # Initialize using similar scheme to nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        
        self.fused_op = fused_conv_scale_min

    def forward(self, x):
        return self.fused_op.fused_conv_scale_min_cuda(x, self.weight, self.bias, self.scale_factor)