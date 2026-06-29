import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_kd1_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kd1_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int H, int W, int D,
    int C_out, int K, int stride, int padding, int dilation, int groups,
    int H_out, int W_out, bool use_bias) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_out * H_out * W_out * D;
    if (idx >= total) return;

    int d = idx % D; idx /= D;
    int w_out = idx % W_out; idx /= W_out;
    int h_out = idx % H_out; idx /= H_out;
    int c_out = idx % C_out; idx /= B;
    int b = idx;

    int c_in_per_group = C_in / groups;
    int c_out_per_group = C_out / groups;
    int group_id = c_out / c_out_per_group;
    int c_in_start = group_id * c_in_per_group;
    
    float sum = 0.0f;
    for (int g_c_in = 0; g_c_in < c_in_per_group; ++g_c_in) {
        int c_in = c_in_start + g_c_in;
        int w_base = c_out * c_in_per_group * K * K + g_c_in * K * K;
        
        int in_offset = b * C_in * H * W * D + c_in * H * W * D + d;
        int in_stride_h = W * D;
        int in_stride_w = D;
        
        for (int kh = 0; kh < K; ++kh) {
            int h_in = h_out * stride - padding + kh * dilation;
            if (h_in < 0 || h_in >= H) continue;
            int row_offset = in_offset + h_in * in_stride_h;
            
            for (int kw = 0; kw < K; ++kw) {
                int w_in = w_out * stride - padding + kw * dilation;
                if (w_in < 0 || w_in >= W) continue;
                sum += input[row_offset + w_in * in_stride_w] * weight[w_base + kh * K + kw];
            }
        }
    }
    if (use_bias) {
        sum += bias[c_out];
    }
    output[idx] = sum;
}

torch::Tensor conv3d_kd1_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                              int stride, int padding, int dilation, int groups, bool use_bias) {
    int B = input.size(0);
    int C_in = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int D = input.size(4);
    int C_out = weight.size(0);
    int K = weight.size(2);
    
    int H_out = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    
    auto output = torch::empty({B, C_out, H_out, W_out, D}, input.options());
    
    const int block_size = 256;
    int total_elements = B * C_out * H_out * W_out * D;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv3d_kd1_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, H, W, D,
        C_out, K, stride, padding, dilation, groups,
        H_out, W_out, use_bias
    );
    
    return output;
}
"""

conv3d_kd1_cpp_source = """
#include <torch/extension.h>
torch::Tensor conv3d_kd1_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
                              int stride, int padding, int dilation, int groups, bool use_bias);
"""

conv3d_kd1 = load_inline(
    name="conv3d_kd1",
    cpp_sources=conv3d_kd1_cpp_source,
    cuda_sources=conv3d_kd1_source,
    functions=["conv3d_kd1_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        self.conv_op = conv3d_kd1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv3d.weight
        bias = self.conv3d.bias if self.use_bias else torch.zeros(self.conv3d.out_channels, device=x.device, dtype=x.dtype)
        return self.conv_op.conv3d_kd1_cuda(x, weight, bias, self.stride, self.padding, self.dilation, self.groups, self.use_bias)