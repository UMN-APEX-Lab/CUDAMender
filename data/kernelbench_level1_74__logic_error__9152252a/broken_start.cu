import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for ConvTranspose1d
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
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool has_bias) {
    
    int batch_idx = blockIdx.x;
    int out_channel = blockIdx.y;
    int out_pos = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_channel >= out_channels || out_pos >= output_length) {
        return;
    }
    
    float val = has_bias ? bias[out_channel] : 0.0f;
    
    // For each input position that contributes to this output position
    for (int in_pos = 0; in_pos < input_length; in_pos++) {
        int out_pos_start = in_pos * stride - padding;
        int out_pos_end = out_pos_start + dilation * (kernel_size - 1);
        
        if (out_pos >= out_pos_start && out_pos <= out_pos_end) {
            if ((out_pos - out_pos_start) % dilation == 0) {
                int kernel_idx = (out_pos - out_pos_start) / dilation;
                if (kernel_idx >= 0 && kernel_idx < kernel_size) {
                    int in_channel = out_channel % in_channels;
                    int weight_idx = out_channel * in_channels * kernel_size + in_channel * kernel_size + kernel_idx;
                    int input_idx = batch_idx * in_channels * input_length + in_channel * input_length + in_pos;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    int output_idx = batch_idx * out_channels * output_length + out_channel * output_length + out_pos;
    output[output_idx] = val;
}

torch::Tensor conv_transpose1d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation) {
    
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto input_length = input.size(2);
    auto out_channels = weight.size(0);
    auto kernel_size = weight.size(2);
    
    int output_length = (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    
    auto output = torch::zeros({batch_size, out_channels, output_length}, 
                                input.options());
    
    bool has_bias = bias.defined() && bias.numel() > 0;
    
    dim3 blocks(batch_size, out_channels, (output_length + 255) / 256);
    dim3 threads(256);
    
    conv_transpose1d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation,
        has_bias
    );
    
    cudaDeviceSynchronize();
    
    return output;
}
"""

conv_transpose1d_cpp_source = (
    "torch::Tensor conv_transpose1d_cuda("
    "torch::Tensor input, "
    "torch::Tensor weight, "
    "torch::Tensor bias, "
    "int stride, "
    "int padding, "
    "int dilation);"
)

# Compile the inline CUDA code
conv_transpose1d = load_inline(
    name="conv_transpose1d",
    cpp_sources=conv_transpose1d_cpp_source,
    cuda_sources=conv_transpose1d_source,
    functions=["conv_transpose1d_cuda"],
    verbose=True,
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size))
        
        if bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias_param', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias_param is not None:
            return conv_transpose1d.conv_transpose1d_cuda(
                x, self.weight, self.bias_param, 
                self.stride, self.padding, self.dilation
            )
        else:
            empty_bias = torch.empty(0, device=x.device, dtype=x.dtype)
            return conv_transpose1d.conv_transpose1d_cuda(
                x, self.weight, empty_bias, 
                self.stride, self.padding, self.dilation
            )