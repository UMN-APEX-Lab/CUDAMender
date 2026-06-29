import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

__device__ __forceinline__ float lrelu(float x, float neg_slope) {
    return x > 0.0f ? x : x * neg_slope;
}

__global__ void fused_act_mul_act_pool3d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ mult,
    float* __restrict__ out,
    int N, int C, int D, int H, int W,
    int Do, int Ho, int Wo,
    float neg_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Do * Ho * Wo;
    if (idx >= total) return;

    int tmp = idx;
    int wo = tmp % Wo; tmp /= Wo;
    int ho = tmp % Ho; tmp /= Ho;
    int do_ = tmp % Do; tmp /= Do;
    int c = tmp % C; tmp /= C;
    int n = tmp;

    int d0 = do_ * 2;
    int h0 = ho * 2;
    int w0 = wo * 2;

    float m = mult[c];
    float maxv = -CUDART_INF_F;

    #pragma unroll
    for (int kd = 0; kd < 2; ++kd) {
        int di = d0 + kd;
        #pragma unroll
        for (int kh = 0; kh < 2; ++kh) {
            int hi = h0 + kh;
            #pragma unroll
            for (int kw = 0; kw < 2; ++kw) {
                int wi = w0 + kw;
                int in_idx = (((n * C + c) * D + di) * H + hi) * W + wi;
                float v = x[in_idx];
                v = lrelu(v, neg_slope);
                v = v * m;
                v = lrelu(v, neg_slope);
                maxv = v > maxv ? v : maxv;
            }
        }
    }

    out[idx] = maxv;
}

torch::Tensor fused_act_mul_act_pool3d_cuda(
    torch::Tensor x,
    torch::Tensor mult,
    double negative_slope
) {
    CHECK_CUDA(x);
    CHECK_CUDA(mult);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(mult);
    CHECK_FLOAT(x);
    CHECK_FLOAT(mult);
    TORCH_CHECK(x.dim() == 5, "x must be a 5D tensor [N, C, D, H, W]");
    TORCH_CHECK(mult.numel() == x.size(1), "multiplier must have numel equal to channels");

    const auto N = (int)x.size(0);
    const auto C = (int)x.size(1);
    const auto D = (int)x.size(2);
    const auto H = (int)x.size(3);
    const auto W = (int)x.size(4);

    TORCH_CHECK(D >= 2 && H >= 2 && W >= 2, "input spatial dims must be >= 2 for MaxPool3d(kernel_size=2)");
    const int Do = D / 2;
    const int Ho = H / 2;
    const int Wo = W / 2;

    auto out = torch::empty({N, C, Do, Ho, Wo}, x.options());

    const int total = N * C * Do * Ho * Wo;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_act_mul_act_pool3d_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(),
        mult.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W, Do, Ho, Wo,
        (float)negative_slope
    );

    return out;
}
"""

_cpp_src = r"""
torch::Tensor fused_act_mul_act_pool3d_cuda(
    torch::Tensor x,
    torch::Tensor mult,
    double negative_slope
);
"""

_fused_ext = load_inline(
    name="fused_act_mul_act_pool3d_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["fused_act_mul_act_pool3d_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model: keeps ConvTranspose3d in PyTorch/cuDNN and fuses
    LeakyReLU -> channelwise multiply -> LeakyReLU -> MaxPool3d(kernel=2,stride=2)
    into a custom CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.negative_slope = 0.2
        self._ext = _fused_ext

    def forward(self, x):
        x = self.conv_transpose(x)
        if x.is_cuda and x.dtype == torch.float32:
            mult = self.multiplier.contiguous().view(-1)
            return self._ext.fused_act_mul_act_pool3d_cuda(x.contiguous(), mult, self.negative_slope)
        x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = x * self.multiplier
        x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = torch.nn.functional.max_pool3d(x, kernel_size=2)
        return x