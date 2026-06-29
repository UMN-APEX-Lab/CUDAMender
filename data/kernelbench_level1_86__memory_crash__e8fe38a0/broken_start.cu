import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused depthwise-separable convolution
depthwise_separable_conv_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>

__global__ void depthwise_separable_conv_kernel(
    const float* input,
    const float* depthwise_weight,
    const float* pointwise_weight,
    const float* depthwise_bias,
    const float* pointwise_bias,
    float* output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool use_depthwise_bias,
    bool use_pointwise_bias
) {
    // Each thread handles one output element
    int total_outputs = batch_size * out_channels * out_height * out_width;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_outputs) {
        return;
    }
    
    // Decode flat index to (batch, out_c, out_h, out_w)
    int out_w = idx % out_width;
    int temp = idx / out_width;
    int out_h = temp % out_height;
    temp = temp / out_height;
    int out_c = temp % out_channels;
    int batch_idx = temp / out_channels;
    
    // Compute depthwise output for this channel and spatial position
    float depthwise_output = 0.0f;
    int in_h_start = out_h * stride - padding;
    int in_w_start = out_w * stride - padding;
    
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int in_h = in_h_start + kh * dilation;
            int in_w = in_w_start + kw * dilation;
            
            if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
                int in_idx = ((batch_idx * in_channels + out_c) * in_height + in_h) * in_width + in_w;
                int dw_weight_idx = (out_c * kernel_size + kh) * kernel_size + kw;
                depthwise_output += input[in_idx] * depthwise_weight[dw_weight_idx];
            }
        }
    }
    
    // Add depthwise bias if present
    if (use_depthwise_bias) {
        depthwise_output += depthwise_bias[out_c];
    }
    
    // Apply pointwise (1x1) convolution
    float pointwise_output = 0.0f;
    for (int ic = 0; ic < in_channels; ic++) {
        int pw_weight_idx = out_c * in_channels + ic;
        pointwise_output += pointwise_weight[pw_weight_idx] * depthwise_output;
    }
    
    // Add pointwise bias if present
    if (use_pointwise_bias) {
        pointwise_output += pointwise_bias[out_c];
    }
    
    output[idx] = pointwise_output;
}

torch::Tensor depthwise_separable_conv_cuda(
    torch::Tensor input,
    torch::Tensor depthwise_weight,
    torch::Tensor pointwise_weight,
    torch::Tensor depthwise_bias,
    torch::Tensor pointwise_bias,
    int stride,
    int padding,
    int dilation,
    bool use_depthwise_bias,
    bool use_pointwise_bias
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    int out_channels = pointwise_weight.size(0);
    
    int kernel_size = depthwise_weight.size(2);
    
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    const int block_size = 256;
    const int num_blocks = (batch_size * out_channels * out_height * out_width + block_size - 1) / block_size;
    
    depthwise_separable_conv_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        depthwise_weight.data_ptr<float>(),
        pointwise_weight.data_ptr<float>(),
        use_depthwise_bias ? depthwise_bias.data_ptr<float>() : nullptr,
        use_pointwise_bias ? pointwise_bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation,
        use_depthwise_bias,
        use_pointwise_bias
    );
    
    cudaDeviceSynchronize();
    
    return output;
}
"""

depthwise_separable_conv_cpp_source = (
    "torch::Tensor depthwise_separable_conv_cuda("
    "torch::Tensor input,"
    "torch::Tensor depthwise_weight,"
    "torch::Tensor pointwise_weight,"
    "torch::Tensor depthwise_bias,"
    "torch::Tensor pointwise_bias,"
    "int stride,"
    "int padding,"
    "int dilation,"
    "bool use_depthwise_bias,"
    "bool use_pointwise_bias"
    ");"
)

# Compile the inline CUDA code for fused depthwise-separable convolution
depthwise_separable_conv = load_inline(
    name="depthwise_separable_conv",
    cpp_sources=depthwise_separable_conv_cpp_source,
    cuda_sources=depthwise_separable_conv_source,
    functions=["depthwise_separable_conv_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized depthwise-separable 2D convolution using fused CUDA kernel.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
                 padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Create learnable parameters matching Conv2d format
        # Depthwise: [in_channels, 1, kernel_size, kernel_size]
        self.depthwise_weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        # Pointwise: [out_channels, in_channels, 1, 1]
        self.pointwise_weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        
        if bias:
            self.depthwise_bias = nn.Parameter(torch.empty(in_channels))
            self.pointwise_bias = nn.Parameter(torch.empty(out_channels))
            self.use_depthwise_bias = True
            self.use_pointwise_bias = True
        else:
            self.register_buffer('depthwise_bias', torch.empty(0))
            self.register_buffer('pointwise_bias', torch.empty(0))
            self.use_depthwise_bias = False
            self.use_pointwise_bias = False
        
        # Initialize weights
        self._initialize_weights()
        
        # Register CUDA operator
        self.depthwise_separable_conv = depthwise_separable_conv

    def _initialize_weights(self):
        # Xavier initialization for depthwise weights
        nn.init.xavier_uniform_(self.depthwise_weight)
        nn.init.xavier_uniform_(self.pointwise_weight)
        
        if self.bias:
            nn.init.zeros_(self.depthwise_bias)
            nn.init.zeros_(self.pointwise_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the fused depthwise-separable 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        return self.depthwise_separable_conv.depthwise_separable_conv_cuda(
            x,
            self.depthwise_weight,
            self.pointwise_weight,
            self.depthwise_bias,
            self.pointwise_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.use_depthwise_bias,
            self.use_pointwise_bias
        )