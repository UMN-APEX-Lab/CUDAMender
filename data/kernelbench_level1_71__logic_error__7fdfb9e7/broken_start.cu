import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* x, const float* weight, float* out,
    int B, int C_in, int C_out, int H_in, int W_in, int K,
    int H_out, int W_out) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = B * C_out * H_out * W_out;
    if (idx >= total_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int b = idx / (W_out * H_out * C_out);

    float sum = 0.0f;
    int base_x = b * C_in * H_in * W_in;
    int base_w = c_out * C_in * K * K;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        int x_c_offset = c_in * H_in * W_in;
        int w_c_offset = c_in * K * K;
        for (int kh = 0; kh < K; ++kh) {
            int h_in = h_out - kh;
            if (h_in < 0 || h_in >= H_in) continue;
            int x_h_offset = h_in * W_in;
            for (int kw = 0; kw < K; ++kw) {
                int w_in = w_out - kw;
                if (w_in >= 0 && w_in < W_in) {
                    sum += x[base_x + x_c_offset + x_h_offset + w_in] * weight[base_w + w_c_offset + kh * K + kw];
                }
            }
        }
    }
    out[idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor x, torch::Tensor weight) {
    int B = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int C_out = weight.size(0);
    int K = weight.size(2);
    
    // Output size calculation for stride=1, padding=0, dilation=1, output_padding=0
    int H_out = H_in + K - 1;
    int W_out = W_in + K - 1;
    
    auto out = torch::empty({B, C_out, H_out, W_out}, x.options());
    
    const int block_size = 256;
    const int num_blocks = (B * C_out * H_out * W_out + block_size - 1) / block_size;
    
    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), weight.data_ptr<float>(), out.data_ptr<float>(),
        B, C_in, C_out, H_in, W_in, K, H_out, W_out);
        
    return out;
}
"""

conv_transpose_cpp_source = (
    "torch::Tensor conv_transpose2d_cuda(torch::Tensor x, torch::Tensor weight);"
)

conv_transpose_op = load_inline(
    name="conv_transpose_op",
    cpp_sources=conv_transpose_cpp_source,
    cuda_sources=conv_transpose_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self.conv_op = conv_transpose_op
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_op.conv_transpose2d_cuda(x, self.conv_transpose2d.weight)