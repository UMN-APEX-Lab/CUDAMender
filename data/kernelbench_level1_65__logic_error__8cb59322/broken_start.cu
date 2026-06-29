import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int KH, int KW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float sum = 0.0f;
    
    int in_col_stride = H_in * W_in;
    int in_row_step = W_in;

    int wt_c_out_stride = KH * KW;
    int wt_c_in_stride = C_out * KH * KW;
    int wt_kh_step = KW;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        const float* in_ptr = input + n * C_in * in_col_stride + c_in * in_col_stride;
        const float* wt_ptr = weight + c_in * wt_c_in_stride + c_out * wt_c_out_stride;
        
        for (int kh = 0; kh < KH; ++kh) {
            int h_in = h_out - kh;
            if (h_in >= 0 && h_in < H_in) {
                for (int kw = 0; kw < KW; ++kw) {
                    int w_in = w_out - kw;
                    if (w_in >= 0 && w_in < W_in) {
                        sum += in_ptr[h_in * in_row_step + w_in] * wt_ptr[kh * wt_kh_step + kw];
                    }
                }
            }
        }
    }
    output[idx] = sum;
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int KH, int KW) {
    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1);
    int H_out = H_in + KH - 1;
    int W_out = W_in + KW - 1;

    auto output = torch::zeros({N, C_out, H_out, W_out}, input.options());

    int total = N * C_out * H_out * W_out;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    conv_transpose2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW
    );

    return output;
}
"""

conv_transpose2d_cpp_source = "torch::Tensor conv_transpose2d_cuda(torch::Tensor input, torch::Tensor weight, int KH, int KW);"

conv_transpose2d_op = load_inline(
    name="conv_transpose2d_op",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"]
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)
        self.KH = kernel_size[0]
        self.KW = kernel_size[1]

    def forward(self, x):
        return conv_transpose2d_op.conv_transpose2d_cuda(x, self.conv_transpose2d.weight, self.KH, self.KW)