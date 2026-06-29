import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv2d_fwd_kernel(
    const float* input,
    const float* weight,
    float* output,
    int N, int C, int H, int W,
    int C_out, int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dil_h, int dil_w,
    int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float sum = 0.0f;
    int base_in_idx = n * C * H * W;
    int base_w_idx = c_out * C * KH * KW;

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            int h_in = h_out * stride_h - pad_h + kh * dil_h;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int w_in = w_out * stride_w - pad_w + kw * dil_w;
                if (w_in >= 0 && w_in < W) {
                    sum += input[base_in_idx + c * H * W + h_in * W + w_in] *
                           weight[base_w_idx + c * KH * KW + kh * KW + kw];
                }
            }
        }
    }
    output[idx] = sum;
}

torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dil_h, int dil_w) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int C_out = weight.size(0);
    int KH = weight.size(2);
    int KW = weight.size(3);

    int H_out = (H + 2 * pad_h - dil_h * (KH - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * pad_w - dil_w * (KW - 1) - 1) / stride_w + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    int total = N * C_out * H_out * W_out;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    conv2d_fwd_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        C_out, KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        dil_h, dil_w,
        H_out, W_out
    );

    return output;
}
"""

conv2d_cpp_source = """
#include <torch/extension.h>
torch::Tensor conv2d_cuda(torch::Tensor input, torch::Tensor weight,
                          int stride_h, int stride_w,
                          int pad_h, int pad_w,
                          int dil_h, int dil_w);
"""

conv2d_op = load_inline(
    name="conv2d_op",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_cuda_source,
    functions=["conv2d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert groups == 1, "Only groups=1 is supported in this optimized kernel."
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = conv2d_op.conv2d_cuda(
            x, self.conv.weight,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )
        if self.bias:
            out = out + self.conv.bias.view(1, -1, 1, 1)
        return out