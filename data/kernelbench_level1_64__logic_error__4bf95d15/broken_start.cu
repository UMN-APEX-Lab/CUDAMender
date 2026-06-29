import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

conv_transpose1d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    bool has_bias
) {
    int batch_idx = blockIdx.y;
    int out_channel = blockIdx.x;
    int group = out_channel / (out_channels / groups);
    int out_channel_in_group = out_channel % (out_channels / groups);
    
    int in_channel_start = group * (in_channels / groups);
    
    int out_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    int tid = threadIdx.x;
    
    for (int out_pos = tid; out_pos < out_length; out_pos += blockDim.x) {
        int output_idx = ((batch_idx * out_channels + out_channel) * out_length + out_pos);
        
        float sum = 0.0f;
        
        for (int k = 0; k < kernel_size; k++) {
            int in_pos = (out_pos - k + padding) / stride;
            
            if (in_pos >= 0 && in_pos < input_length) {
                if ((out_pos - k + padding) % stride == 0) {
                    int in_idx = ((batch_idx * in_channels + in_channel_start) * input_length + in_pos);
                    
                    for (int c = 0; c < in_channels / groups; c++) {
                        int in_channel = in_channel_start + c;
                        int weight_idx = ((out_channel_in_group * (in_channels / groups) + c) * kernel_size + k);
                        sum += input[in_idx] * weight[weight_idx];
                    }
                }
            }
        }
        
        if (has_bias) {
            sum += bias[out_channel];
        }
        
        output[output_idx] = sum;
    }
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int out_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::empty({batch_size, out_channels, out_length}, input.options());
    
    int threads_per_block = 256;
    dim3 blocks(out_channels, batch_size);
    dim3 threads(threads_per_block);
    
    conv_transpose1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        kernel_size,
        stride,
        padding,
        output_padding,
        groups,
        bias.numel() > 0
    );
    
    return output;
}
"""

conv_transpose1d_cpp_source = (
    "torch::Tensor conv_transpose1d_cuda("
    "    torch::Tensor input,"
    "    torch::Tensor weight,"
    "    torch::Tensor bias,"
    "    int stride,"
    "    int padding,"
    "    int output_padding,"
    "    int groups"
    ");"
)

conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.bias_flag = bias
        
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_tensor = self.bias if self.bias is not None else torch.empty(0, dtype=x.dtype, device=x.device)
        
        return conv_transpose1d.conv_transpose1d_cuda(
            x,
            self.weight,
            bias_tensor,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups
        )