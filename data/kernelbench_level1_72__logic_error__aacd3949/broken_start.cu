import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* x, const float* w, float* y,
    int N, int C_in, int C_out, int D_in, int H_in, int W_in,
    int K_d, int K_h, int K_w,
    int S_d, int S_h, int S_w,
    int P_d, int P_h, int P_w,
    int OP_d, int OP_h, int OP_w,
    int G) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int D_out = (D_in - 1) * S_d - 2 * P_d + K_d + OP_d;
    int H_out = (H_in - 1) * S_h - 2 * P_h + K_h + OP_h;
    int W_out = (W_in - 1) * S_w - 2 * P_w + K_w + OP_w;
    int total_out = N * C_out * D_out * H_out * W_out;
    
    if (out_idx >= total_out) return;

    int tmp = out_idx;
    int w_out = tmp % W_out; tmp /= W_out;
    int h_out = tmp % H_out; tmp /= H_out;
    int d_out = tmp % D_out; tmp /= D_out;
    int c_out = tmp % C_out; tmp /= C_out;
    int n = tmp;

    int C_out_per_G = C_out / G;
    int C_in_per_G = C_in / G;
    int group = c_out / C_out_per_G;
    int c_in_start = group * C_in_per_G;
    int c_in_end = c_in_start + C_in_per_G;
    int c_out_local = c_out % C_out_per_G;

    float sum = 0.0f;
    for (int c_in = c_in_start; c_in < c_in_end; ++c_in) {
        for (int kd = 0; kd < K_d; ++kd) {
            int num_d = d_out + 2 * P_d - kd - OP_d + S_d;
            if (num_d < 0 || num_d % S_d != 0) continue;
            int d_in = num_d / S_d;
            if (d_in >= D_in) continue;

            for (int kh = 0; kh < K_h; ++kh) {
                int num_h = h_out + 2 * P_h - kh - OP_h + S_h;
                if (num_h < 0 || num_h % S_h != 0) continue;
                int h_in = num_h / S_h;
                if (h_in >= H_in) continue;

                for (int kw = 0; kw < K_w; ++kw) {
                    int num_w = w_out + 2 * P_w - kw - OP_w + S_w;
                    if (num_w < 0 || num_w % S_w != 0) continue;
                    int w_in = num_w / S_w;
                    if (w_in >= W_in) continue;

                    float x_val = x[n * C_in * D_in * H_in * W_in + 
                                    c_in * D_in * H_in * W_in + 
                                    d_in * H_in * W_in + 
                                    h_in * W_in + w_in];
                    float w_val = w[c_in * C_out_per_G * K_d * K_h * K_w + 
                                    c_out_local * K_d * K_h * K_w + 
                                    kd * K_h * K_w + 
                                    kh * K_w + kw];
                    sum += x_val * w_val;
                }
            }
        }
    }
    y[out_idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor x, torch::Tensor w,
                                    int S_d, int S_h, int S_w,
                                    int P_d, int P_h, int P_w,
                                    int OP_d, int OP_h, int OP_w,
                                    int G) {
    auto N = x.size(0);
    auto C_in = x.size(1);
    auto D_in = x.size(2);
    auto H_in = x.size(3);
    auto W_in = x.size(4);
    auto C_out = w.size(0);
    auto K_d = w.size(2);
    auto K_h = w.size(3);
    auto K_w = w.size(4);

    int D_out = (D_in - 1) * S_d - 2 * P_d + K_d + OP_d;
    int H_out = (H_in - 1) * S_h - 2 * P_h + K_h + OP_h;
    int W_out = (W_in - 1) * S_w - 2 * P_w + K_w + OP_w;

    auto y = torch::zeros({N, C_out, D_out, H_out, W_out}, x.options());

    const int block_size = 256;
    const int num_blocks = (y.numel() + block_size - 1) / block_size;

    conv_transpose3d_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), w.data_ptr<float>(), y.data_ptr<float>(),
        N, C_in, C_out, D_in, H_in, W_in,
        K_d, K_h, K_w,
        S_d, S_h, S_w,
        P_d, P_h, P_w,
        OP_d, OP_h, OP_w,
        G);
        
    return y;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor conv_transpose3d_cuda(torch::Tensor x, torch::Tensor w,
                                    int S_d, int S_h, int S_w,
                                    int P_d, int P_h, int P_w,
                                    int OP_d, int OP_h, int OP_w,
                                    int G);
"""

conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), output_padding=(0,0,0), groups=1, bias=False):
        super().__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups

    def forward(self, x):
        w = self.conv_transpose3d.weight
        return conv_transpose3d_op.conv_transpose3d_cuda(
            x, w,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.output_padding[0], self.output_padding[1], self.output_padding[2],
            self.groups
        )