import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int D, int H, int W,
    int K, int stride, int padding, int dilation,
    int D_out, int H_out, int W_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int d_out = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / (W_out * H_out * D_out)) % C_out;
    int n = idx / (W_out * H_out * D_out * C_out);

    float sum = 0.0f;
    int weight_base = c_out * C_in * K * K * K;
    int input_base = n * C_in * D * H * W;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        int input_c_base = input_base + c_in * D * H * W;
        int weight_c_base = weight_base + c_in * K * K * K;
        for (int kd = 0; kd < K; ++kd) {
            int d_in = d_out * stride + kd * dilation - padding;
            if (d_in < 0 || d_in >= D) continue;
            int input_d_base = input_c_base + d_in * H * W;
            int weight_kd_base = weight_c_base + kd * K * K;
            for (int kh = 0; kh < K; ++kh) {
                int h_in = h_out * stride + kh * dilation - padding;
                if (h_in < 0 || h_in >= H) continue;
                int input_h_base = input_d_base + h_in * W;
                int weight_kh_base = weight_kd_base + kh * K;
                for (int kw = 0; kw < K; ++kw) {
                    int w_in = w_out * stride + kw * dilation - padding;
                    if (w_in >= 0 && w_in < W) {
                        sum += input[input_h_base + w_in] * weight[weight_kh_base + kw];
                    }
                }
            }
        }
    }
    output[idx] = sum;
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation) {
    int N = input.size(0);
    int C_in = input.size(1);
    int D = input.size(2);
    int H = input.size(3);
    int W = input.size(4);
    int C_out = weight.size(0);
    int K = weight.size(2);

    int D_out = (D + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int H_out = (H + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - dilation * (K - 1) - 1) / stride + 1;

    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    int total = N * C_out * D_out * H_out * W_out;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    conv3d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, C_out, D, H, W, K, stride, padding, dilation, D_out, H_out, W_out
    );

    return output;
}
"""

cpp_source = "torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, int stride, int padding, int dilation);"

conv3d_op = load_inline(
    name="conv3d_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv3d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        assert groups == 1, "Only groups=1 is supported in this custom kernel."
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = conv3d_op.conv3d_cuda(x, self.weight, self.stride, self.padding, self.dilation)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        return out