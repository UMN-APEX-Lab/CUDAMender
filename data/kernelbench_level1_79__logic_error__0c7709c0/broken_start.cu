import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ConvTranspose1d is a complex operation. For a custom CUDA implementation to be 
# competitive with cuDNN (which PyTorch uses), we would need to implement a 
# highly optimized GEMM-based approach or a direct convolution kernel. 
# Given the constraints and the nature of the task, the most efficient way to 
# "optimize" while ensuring correctness and stability is to leverage 
# PyTorch's optimized native implementation. 
# However, to follow the requirement of providing a "custom CUDA operator" 
# framework for a specific architectural replacement, we can implement a 
# fused kernel if there were element-wise operations. 
# Since ConvTranspose1d is a monolithic primitive, the standard optimization 
# is to ensure it uses the fastest backend.

# To satisfy the request for a custom CUDA implementation for this specific 
# operation, we implement a basic CUDA kernel for the transposed convolution.
# Note: A production-grade ConvTranspose1d is extremely complex; 
# the following is a functional CUDA implementation of the operation.

conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels, int length_in,
    int kernel_size, int stride, int padding, int dilation, int length_out) 
{
    int b = blockIdx.z;
    int oc = blockIdx.y;
    int lw = blockIdx.x * blockDim.x + threadIdx.x;

    if (lw >= length_out) return;

    float val = 0.0f;
    if (bias != nullptr) {
        val = bias[oc];
    }

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            // The relationship for ConvTranspose1d:
            // output_idx = input_idx * stride + k - padding
            // So, input_idx = (output_idx + padding - k) / stride
            int input_idx_num = lw + padding - k * dilation;
            if (input_idx_num >= 0 && input_idx_num % stride == 0) {
                int input_idx = input_idx_num / stride;
                if (input_idx < length_in) {
                    // weight shape: (in_channels, out_channels, kernel_size)
                    int weight_idx = ic * (out_channels * kernel_size) + oc * kernel_size + k;
                    int input_ptr_idx = b * (in_channels * length_in) + ic * length_in + input_idx;
                    val += input[input_ptr_idx] * weight[weight_idx];
                }
            }
        }
    }
    output[b * (out_channels * length_out) + oc * length_out + lw] = val;
}

torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                                   int stride, int padding, int dilation) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int length_in = input.size(2);
    const int out_channels = weight.size(1);
    const int kernel_size = weight.size(2);

    const int length_out = (length_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1;
    auto output = torch::empty({batch_size, out_channels, length_out}, input.options());

    const int block_size = 256;
    const int grid_x = (length_out + block_size - 1) / block_size;
    dim3 grid(grid_x, out_channels, batch_size);
    dim3 block(block_size);

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    conv_transpose1d_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size, in_channels, out_channels, length_in,
        kernel_size, stride, padding, dilation, length_out
    );

    return output;
}
"""

conv_transpose_cpp_source = "torch::Tensor conv_transpose1d_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation);"

conv_transpose_lib = load_inline(
    name="conv_transpose_lib",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose1d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # We keep the weights and bias in a ConvTranspose1d layer to handle initialization and parameter management
        self.conv_layer = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the custom CUDA kernel
        # ConvTranspose1d weight shape is (in_channels, out_channels, kernel_size)
        # which matches our kernel's expected layout.
        return conv_transpose_lib.conv_transpose1d_cuda(
            x.contiguous(), 
            self.conv_layer.weight.contiguous(), 
            self.conv_layer.bias if self.conv_layer.bias is not None else torch.Tensor(), 
            self.stride, 
            self.padding, 
            self.dilation
        )