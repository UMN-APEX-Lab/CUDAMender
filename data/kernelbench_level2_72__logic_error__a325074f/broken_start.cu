import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_bn_avgpool_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_bn_avgpool_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* out,
    int N, int C, int D_in, int H_in, int W_in,
    int D_out, int H_out, int W_out,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * D_out * H_out * W_out;
    if (idx >= total_out) return;

    int w = idx % W_out; idx /= W_out;
    int h = idx % H_out; idx /= H_out;
    int d = idx % D_out; idx /= D_out;
    int c = idx % C; idx /= N;
    int n = idx;

    float sum = 0.0f;
    int count = 0;

    float mean = running_mean[c];
    float var = running_var[c];
    float w_c = weight[c];
    float b_c = bias[c];
    float inv_std = rsqrtf(var + eps);

    int d_start = d * 4;
    int h_start = h * 4;
    int w_start = w * 4;
    
    int d_end = min(d_start + 4, D_in);
    int h_end = min(h_start + 4, H_in);
    int w_end = min(w_start + 4, W_in);

    for (int di = d_start; di < d_end; ++di) {
        for (int hi = h_start; hi < h_end; ++hi) {
            for (int wi = w_start; wi < w_end; ++wi) {
                int in_idx = ((n * C + c) * D_in + di) * H_in * W_in + hi * W_in + wi;
                float val = x[in_idx];
                float bn_val = (val - mean) * inv_std * w_c + b_c;
                sum += bn_val;
                count++;
            }
        }
    }

    out[idx] = sum / count;
}

torch::Tensor fused_bn_avgpool_cuda(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {
    
    int N = x.size(0);
    int C = x.size(1);
    int D_in = x.size(2);
    int H_in = x.size(3);
    int W_in = x.size(4);

    int D_mid = (D_in - 2) / 2 + 1;
    int H_mid = (H_in - 2) / 2 + 1;
    int W_mid = (W_in - 2) / 2 + 1;

    int D_out = (D_mid - 2) / 2 + 1;
    int H_out = (H_mid - 2) / 2 + 1;
    int W_out = (W_mid - 2) / 2 + 1;

    auto out = torch::empty({N, C, D_out, H_out, W_out}, x.options());

    int total_out = N * C * D_out * H_out * W_out;
    int block_size = 256;
    int num_blocks = (total_out + block_size - 1) / block_size;

    fused_bn_avgpool_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        eps
    );

    return out;
}
"""

fused_bn_avgpool_cpp_source = (
    "torch::Tensor fused_bn_avgpool_cuda("
    "torch::Tensor x, torch::Tensor weight, torch::Tensor bias, "
    "torch::Tensor running_mean, torch::Tensor running_var, float eps);"
)

fused_bn_avgpool = load_inline(
    name="fused_bn_avgpool",
    cpp_sources=fused_bn_avgpool_cpp_source,
    cuda_sources=fused_bn_avgpool_source,
    functions=["fused_bn_avgpool_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.fused_bn_avgpool = fused_bn_avgpool

    def forward(self, x):
        x = self.conv_transpose(x)
        weight = self.batch_norm.weight.to(x.device)
        bias = self.batch_norm.bias.to(x.device)
        running_mean = self.batch_norm.running_mean.to(x.device)
        running_var = self.batch_norm.running_var.to(x.device)
        x = self.fused_bn_avgpool.fused_bn_avgpool_cuda(
            x, weight, bias, running_mean, running_var, self.batch_norm.eps
        )
        return x