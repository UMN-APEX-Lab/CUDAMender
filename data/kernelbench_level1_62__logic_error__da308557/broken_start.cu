import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for 2D convolution
conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int dilation_height,
    int dilation_width,
    int groups,
    bool has_bias
) {
    int batch_idx = blockIdx.z;
    int out_channel = blockIdx.y * blockDim.y + threadIdx.y;
    int out_y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || out_channel >= out_channels || 
        out_y >= out_height * out_width) {
        return;
    }
    
    int out_x = out_y % out_width;
    out_y = out_y / out_width;
    
    int group_size = in_channels / groups;
    int group_idx = out_channel / (out_channels / groups);
    
    float acc = 0.0f;
    
    for (int ic = 0; ic < group_size; ic++) {
        int in_channel = group_idx * group_size + ic;
        
        for (int ky = 0; ky < kernel_height; ky++) {
            for (int kx = 0; kx < kernel_width; kx++) {
                int in_y = out_y * stride_height - padding_height + ky * dilation_height;
                int in_x = out_x * stride_width - padding_width + kx * dilation_width;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int in_idx = ((batch_idx * in_channels + in_channel) * in_height + in_y) * in_width + in_x;
                    int weight_idx = ((out_channel * group_size + ic) * kernel_height + ky) * kernel_width + kx;
                    acc += input[in_idx] * weight[weight_idx];
                }
            }
        }
    }
    
    if (has_bias) {
        acc += bias[out_channel];
    }
    
    int out_idx = ((batch_idx * out_channels + out_channel) * out_height + out_y) * out_width + out_x;
    output[out_idx] = acc;
}

torch::Tensor conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    bool has_bias
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_height = weight.size(2);
    int kernel_width = weight.size(3);
    
    int out_height = (in_height + 2 * pad_h - dilation_h * (kernel_height - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * pad_w - dilation_w * (kernel_width - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    dim3 threads(16, 16);
    dim3 blocks((out_height * out_width + threads.x - 1) / threads.x,
                (out_channels + threads.y - 1) / threads.y,
                batch_size);
    
    conv2d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_height,
        kernel_width,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        groups,
        has_bias
    );
    
    cudaDeviceSynchronize();
    
    return output;
}
"""

conv2d_cpp_source = (
    "torch::Tensor conv2d_cuda("
    "    torch::Tensor input,"
    "    torch::Tensor weight,"
    "    torch::Tensor bias,"
    "    int stride_h,"
    "    int stride_w,"
    "    int pad_h,"
    "    int pad_w,"
    "    int dilation_h,"
    "    int dilation_w,"
    "    int groups,"
    "    bool has_bias"
    ");"
)

conv2d = load_inline(
    name="conv2d_custom",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_source,
    functions=["conv2d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.has_bias = bias
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        if self.has_bias:
            bias_tensor = self.bias
        else:
            bias_tensor = torch.empty(0, device=x.device, dtype=x.dtype)
        
        return conv2d.conv2d_cuda(
            x,
            self.weight,
            bias_tensor,
            self.stride[0],
            self.stride[1],
            self.padding[0],
            self.padding[1],
            self.dilation[0],
            self.dilation[1],
            self.groups,
            self.has_bias
        )