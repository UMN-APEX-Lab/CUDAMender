import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv_transpose2d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose2d_kernel(
    const float* x, const float* weight, const float* bias, float* out,
    int B, int IC, int OC, int H_in, int W_in, int H_out, int W_out,
    int K, int stride, int padding, int dilation, bool use_bias) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * OC * H_out * W_out;
    int stride_grid = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride_grid) {
        int temp = i;
        int w_out = temp % W_out; temp /= W_out;
        int h_out = temp % H_out; temp /= H_out;
        int oc = temp % OC; temp /= OC;
        int b = temp;

        double sum = 0.0;

        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int h_in_num = h_out + padding - kh * dilation;
                int w_in_num = w_out + padding - kw * dilation;

                if (h_in_num >= 0 && w_in_num >= 0 &&
                    h_in_num % stride == 0 && w_in_num % stride == 0) {
                    int h_in = h_in_num / stride;
                    int w_in = w_in_num / stride;

                    if (h_in < H_in && w_in < W_in) {
                        for (int ic = 0; ic < IC; ++ic) {
                            int x_idx = ((b * IC + ic) * H_in + h_in) * W_in + w_in;
                            int w_idx = ((ic * OC + oc) * K + kh) * K + kw;
                            sum += (double)x[x_idx] * (double)weight[w_idx];
                        }
                    }
                }
            }
        }
        if (use_bias) {
            sum += (double)bias[oc];
        }
        out[i] = (float)sum;
    }
}

torch::Tensor conv_transpose2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation) {
    x = x.contiguous();
    weight = weight.contiguous();
    if (bias.defined()) bias = bias.contiguous();
    
    int B = x.size(0);
    int IC = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int OC = weight.size(1);
    int K = weight.size(2);

    int H_out = (H_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1;
    int W_out = (W_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1;

    auto out = torch::empty({B, OC, H_out, W_out}, x.options());
    bool use_bias = bias.defined();

    const int block_size = 256;
    int total_elements = B * OC * H_out * W_out;
    int num_blocks = (total_elements + block_size - 1) / block_size;

    if (num_blocks > 0) {
        conv_transpose2d_kernel<<<num_blocks, block_size>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            use_bias ? bias.data_ptr<float>() : nullptr,
            out.data_ptr<float>(),
            B, IC, OC, H_in, W_in, H_out, W_out, K, stride, padding, dilation, use_bias
        );
    }

    return out;
}
"""

conv_transpose2d_cpp_source = """
#include <torch/extension.h>
torch::Tensor conv_transpose2d_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int stride, int padding, int dilation);
"""

conv_transpose2d_op = load_inline(
    name="conv_transpose2d_op",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_cuda_source,
    functions=["conv_transpose2d_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv_transpose2d.weight
        bias = self.conv_transpose2d.bias if self.conv_transpose2d.bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)
        return conv_transpose2d_op.conv_transpose2d_cuda(x, weight, bias, self.stride, self.padding, self.dilation)