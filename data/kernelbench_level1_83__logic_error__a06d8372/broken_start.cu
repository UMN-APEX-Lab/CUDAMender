import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for depthwise convolution with kernel_size x 1 (asymmetric kernel)
# This is a specialized kernel that handles the case where kernel width is 1
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int stride,
    int padding,
    int dilation,
    bool has_bias,
    const float* __restrict__ bias
) {
    // Each thread handles one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * in_channels * out_height * out_width;
    
    if (idx >= total_elements) return;
    
    // Decode index
    int tmp = idx;
    int w = tmp % out_width;
    tmp /= out_width;
    int h = tmp % out_height;
    tmp /= out_height;
    int c = tmp % in_channels;
    int b = tmp / in_channels;
    
    // Compute input start position
    int in_h_start = h * stride - padding;
    int in_w_start = w * stride - padding;
    
    float sum = 0.0f;
    
    // Kernel is (kernel_h, 1) shape
    for (int kh = 0; kh < kernel_h; kh++) {
        int in_h = in_h_start + kh * dilation;
        int in_w = in_w_start; // kernel width is 1, so no loop needed
        
        if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) {
            int input_idx = ((b * in_channels + c) * in_height + in_h) * in_width + in_w;
            int weight_idx = (c * kernel_h + kh); // weight shape: (in_channels, 1, kernel_h, 1)
            sum += input[input_idx] * weight[weight_idx];
        }
    }
    
    if (has_bias) {
        sum += bias[c];
    }
    
    int output_idx = ((b * in_channels + c) * out_height + h) * out_width + w;
    output[output_idx] = sum;
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
) {
    // input: (batch_size, in_channels, height, width)
    // weight: (in_channels, 1, kernel_h, 1) for depthwise
    auto batch_size = input.size(0);
    auto in_channels = input.size(1);
    auto in_height = input.size(2);
    auto in_width = input.size(3);
    
    auto kernel_h = weight.size(2);
    
    // Calculate output dimensions
    auto out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (1 - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, in_channels, out_height, out_width}, input.options());
    
    int total_elements = batch_size * in_channels * out_height * out_width;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    bool has_bias = bias.has_value();
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;
    
    // Reshape weight to (in_channels, kernel_h) for easier access
    auto weight_reshaped = weight.view({in_channels, kernel_h});
    
    depthwise_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight_reshaped.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_h,
        stride,
        padding,
        dilation,
        has_bias,
        bias_ptr
    );
    
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride,
    int padding,
    int dilation
);
"""

# Compile the inline CUDA code for depthwise convolution
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution with custom CUDA kernel.
    Kernel is asymmetric: (kernel_size, 1)
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Weight shape: (in_channels, 1, kernel_size, 1) for depthwise conv
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(in_channels))
        else:
            self.register_parameter('bias', None)
        
        self.depthwise_conv2d = depthwise_conv2d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the optimized depthwise 2D convolution.
        """
        # Ensure input is contiguous and on CUDA
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        
        # Use custom CUDA kernel
        return self.depthwise_conv2d.depthwise_conv2d_cuda(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation
        )

def get_inputs():
    batch_size = 64
    in_channels = 8
    height = 512
    width = 512
    x = torch.rand(batch_size, in_channels, height, width).cuda()
    return [x]

def get_init_inputs():
    in_channels = 8
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    return [in_channels, kernel_size, stride, padding, dilation]