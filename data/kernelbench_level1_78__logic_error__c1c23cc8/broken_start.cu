import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ConvTranspose2d is a complex operation. For general cases, 
# PyTorch's cuDNN implementation is highly optimized.
# However, to fulfill the requirement of a custom CUDA operator 
# while maintaining correctness and reasonable performance, 
# we can implement a fused kernel or a specialized wrapper.
# Since a full cuDNN-level ConvTranspose2d implementation in raw CUDA 
# is thousands of lines of code, we will use a high-performance 
# approach using PyTorch's internal mechanisms or a custom 
# implementation if applicable. 
# For the given asymmetric kernel (3, 7) and stride (1, 1), 
# we can implement a custom kernel that optimizes the specific 
# memory access pattern of the transposed convolution.

conv_transpose_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int in_h, int in_w, int k_h, int k_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int out_h, int out_w) 
{
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int oh = blockIdx.x * blockDim.x + threadIdx.x;

    if (oh >= out_h) return;

    // Output index for the current thread
    // output shape: [batch, out_channels, out_h, out_w]
    
    for (int ow = 0; ow < out_w; ++ow) {
        float val = (bias != nullptr) ? bias[oc] : 0.0f;
        
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < k_h; ++kh) {
                int ih = (oh + pad_h - kh);
                if (ih < 0 || ih % stride_h != 0) continue;
                ih /= stride_h;
                if (ih < 0 || ih >= in_h) continue;

                for (int kw = 0; kw < k_w; ++kw) {
                    int iw = (ow + pad_w - kw);
                    if (iw < 0 || iw % stride_w != 0) continue;
                    iw /= stride_w;
                    if (iw < 0 || iw >= in_w) continue;

                    // weight shape: [in_channels, out_channels, k_h, k_w]
                    // input shape: [batch, in_channels, in_h, in_w]
                    int weight_idx = ic * (out_channels * k_h * k_w) + oc * (k_h * k_w) + kh * k_w + kw;
                    int input_idx = b * (in_channels * in_h * in_w) + ic * (in_h * in_w) + ih * in_w + iw;
                    
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
        int out_idx = b * (out_channels * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow;
        output[out_idx] = val;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                  int stride_h, int stride_w, int pad_h, int pad_w) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_h = input.size(2);
    const int in_w = input.size(3);
    
    const int out_channels = weight.size(1);
    const int k_h = weight.size(2);
    const int k_w = weight.size(3);
    
    const int out_h = (in_h - 1) * stride_h - 2 * pad_h + k_h;
    const int out_w = (in_w - 1) * stride_w - 2 * pad_w + k_w;
    
    auto output = torch::empty({batch_size, out_channels, out_h, out_w}, input.options());
    
    const int block_size = 256;
    const int num_blocks_x = (out_h + block_size - 1) / block_size;
    
    dim3 grid(num_blocks_x, out_channels, batch_size);
    dim3 block(block_size);
    
    float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    conv_transpose2d_kernel<<<grid, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr, output.data_ptr<float>(),
        batch_size, in_channels, out_channels, in_h, in_w, k_h, k_w,
        stride_h, stride_w, pad_h, pad_w, out_h, out_w
    );
    
    return output;
}
"""

conv_transpose_cpp_source = "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride_h, int stride_w, int pad_h, int pad_w);"

conv_transpose_lib = load_inline(
    name="conv_transpose_lib",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Use PyTorch parameters to maintain learnability
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are contiguous and on CUDA
        x = x.contiguous()
        weight = self.weight.contiguous()
        bias = self.bias.contiguous() if self.bias is not None else torch.Tensor()
        
        return conv_transpose_lib.conv_transpose2d_cuda(
            x, weight, bias, 
            self.stride[0], self.stride[1], 
            self.padding[0], self.padding[1]
        )