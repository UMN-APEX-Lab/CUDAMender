import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for depthwise convolution
# Depthwise convolution is essentially a grouped convolution where groups == in_channels.
# Each input channel is convolved with its own filter.
depthwise_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int channels, int height_in, int width_in,
    int height_out, int width_out,
    int kernel_size, int stride, int padding) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * channels * height_out * width_out;

    if (idx < total_elements) {
        int w_out = idx % width_out;
        int h_out = (idx / width_out) % height_out;
        int c = (idx / (width_out * height_out)) % channels;
        int b = idx / (width_out * height_out * channels);

        int h_start = h_out * stride - padding;
        int w_start = w_out * stride - padding;

        float sum = 0.0f;
        for (int kh = 0; kh < kernel_size; ++kh) {
            for (int kw = 0; kw < kernel_size; ++kw) {
                int h_in = h_start + kh;
                int w_in = w_start + kw;

                if (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in) {
                    // input: [B, C, H, W]
                    // weight: [C, 1, KH, KW]
                    int input_idx = ((b * channels + c) * height_in + h_in) * width_in + w_in;
                    int weight_idx = ((c * 1 + 0) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }

        if (bias != nullptr) {
            sum += bias[c];
        }
        output[idx] = sum;
    }
}

torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding) {
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int height_in = input.size(2);
    const int width_in = input.size(3);
    const int kernel_size = weight.size(2);
    
    const int height_out = (height_in + 2 * padding - kernel_size) / stride + 1;
    const int width_out = (width_in + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, height_out, width_out}, input.options());

    int total_elements = batch_size * channels * height_out * width_out;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    depthwise_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size, channels, height_in, width_in,
        height_out, width_out,
        kernel_size, stride, padding
    );

    return output;
}
"""

depthwise_conv_cpp_source = (
    "torch::Tensor depthwise_conv2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding);"
)

depthwise_conv_op = load_inline(
    name="depthwise_conv",
    cpp_sources=depthwise_conv_cpp_source,
    cuda_sources=depthwise_conv_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        # Depthwise conv: groups == in_channels. 
        # For standard depthwise, out_channels usually equals in_channels.
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(in_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is contiguous for the CUDA kernel
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else torch.Tensor()
        
        return depthwise_conv_op.depthwise_conv2d_cuda(x, weight, bias, self.stride, self.padding)