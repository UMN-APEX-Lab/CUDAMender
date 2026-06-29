import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

transposed_conv_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void transposed_conv2d_kernel(
    const float* input,
    const float* weight,
    float* output,
    int B, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int K_h, int K_w,
    int S_h, int S_w,
    int P_h, int P_w,
    int D_h, int D_w,
    int groups
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = B * C_out * H_out * W_out;
    if (idx >= total_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int b = idx / (W_out * H_out * C_out);

    int out_ch_per_g = C_out / groups;
    int in_ch_per_g = C_in / groups;
    int g = c_out / out_ch_per_g;
    int c_out_local = c_out % out_ch_per_g;

    float sum = 0.0f;

    for (int kh = 0; kh < K_h; ++kh) {
        int h_num = h_out + 2 * P_h - D_h * kh;
        if (h_num < 0 || h_num % S_h != 0) continue;
        int h_in = h_num / S_h;
        if (h_in >= H_in) continue;

        for (int kw = 0; kw < K_w; ++kw) {
            int w_num = w_out + 2 * P_w - D_w * kw;
            if (w_num < 0 || w_num % S_w != 0) continue;
            int w_in = w_num / S_w;
            if (w_in >= W_in) continue;

            for (int c_in_local = 0; c_in_local < in_ch_per_g; ++c_in_local) {
                int c_in = g * in_ch_per_g + c_in_local;
                int in_idx = ((b * C_in + c_in) * H_in + h_in) * W_in + w_in;
                int w_idx = ((c_in * out_ch_per_g + c_out_local) * K_h + kh) * K_w + kw;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }

    int out_idx = ((b * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[out_idx] = sum;
}

torch::Tensor transposed_conv2d_cuda(torch::Tensor input, torch::Tensor weight, 
                                     int stride_h, int stride_w, 
                                     int padding_h, int padding_w, 
                                     int dilation_h, int dilation_w, 
                                     int groups) {
    int B = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    
    int out_ch_per_g = weight.size(1);
    int K_h = weight.size(2);
    int K_w = weight.size(3);
    int C_out = out_ch_per_g * groups;
    
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + dilation_h * (K_h - 1) + 1;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + dilation_w * (K_w - 1) + 1;
    
    auto output = torch::empty({B, C_out, H_out, W_out}, input.options());
    
    int total_out = B * C_out * H_out * W_out;
    int block_size = 256;
    int num_blocks = (total_out + block_size - 1) / block_size;
    
    transposed_conv2d_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C_in, C_out, H_in, W_in, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups
    );
    
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>
torch::Tensor transposed_conv2d_cuda(torch::Tensor input, torch::Tensor weight, 
                                     int stride_h, int stride_w, 
                                     int padding_h, int padding_w, 
                                     int dilation_h, int dilation_w, 
                                     int groups);
"""

transposed_conv = load_inline(
    name="transposed_conv",
    cpp_sources=cpp_source,
    cuda_sources=transposed_conv_cuda_source,
    functions=["transposed_conv2d_cuda"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1), padding: tuple = (0, 0), dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super().__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = self.conv_transpose2d.bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = transposed_conv.transposed_conv2d_cuda(
            x, 
            self.conv_transpose2d.weight, 
            self.stride[0], self.stride[1], 
            self.padding[0], self.padding[1], 
            self.dilation[0], self.dilation[1], 
            self.groups
        )
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out