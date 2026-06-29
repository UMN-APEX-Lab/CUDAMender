import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cpp_source = r"""
#include <torch/extension.h>

torch::Tensor fused_residual_logsumexp_cuda(torch::Tensor x_conv, torch::Tensor x_norm);

torch::Tensor fused_residual_logsumexp(torch::Tensor x_conv, torch::Tensor x_norm) {
    if (!x_conv.is_cuda()) {
        auto z = torch::tanh(x_norm);
        auto hs = z * torch::clamp(z + 3.0, 0.0, 6.0) / 6.0;
        auto x_res = x_conv + hs;
        return torch::logsumexp(x_res, 1, true);
    }
    return fused_residual_logsumexp_cuda(x_conv, x_norm);
}
"""

_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

__device__ __forceinline__ float hard_swish_from_tanh(float x) {
    float t = tanhf(x);
    float v = t + 3.0f;
    v = v < 0.0f ? 0.0f : (v > 6.0f ? 6.0f : v);
    return t * v * (1.0f / 6.0f);
}

__global__ void fused_residual_logsumexp_kernel(
    const float* __restrict__ x_conv,
    const float* __restrict__ x_norm,
    float* __restrict__ out,
    int N, int C, int H, int W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = N * H * W;
    if (idx >= spatial) return;

    int hw = idx % (H * W);
    int n = idx / (H * W);
    int h = hw / W;
    int w = hw % W;

    int base = ((n * C) * H + h) * W + w;
    int stride_c = H * W;

    float max_val = -CUDART_INF_F;
    for (int c = 0; c < C; ++c) {
        int off = base + c * stride_c;
        float v = x_conv[off] + hard_swish_from_tanh(x_norm[off]);
        max_val = v > max_val ? v : max_val;
    }

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        int off = base + c * stride_c;
        float v = x_conv[off] + hard_swish_from_tanh(x_norm[off]);
        sum += expf(v - max_val);
    }

    out[idx] = logf(sum) + max_val;
}

torch::Tensor fused_residual_logsumexp_cuda(torch::Tensor x_conv, torch::Tensor x_norm) {
    CHECK_CUDA(x_conv);
    CHECK_CUDA(x_norm);
    CHECK_CONTIGUOUS(x_conv);
    CHECK_CONTIGUOUS(x_norm);
    CHECK_FLOAT(x_conv);
    CHECK_FLOAT(x_norm);
    TORCH_CHECK(x_conv.sizes() == x_norm.sizes(), "x_conv and x_norm must have same shape");
    TORCH_CHECK(x_conv.dim() == 4, "expected 4D tensors");

    const auto N = (int)x_conv.size(0);
    const auto C = (int)x_conv.size(1);
    const auto H = (int)x_conv.size(2);
    const auto W = (int)x_conv.size(3);

    auto out = torch::empty({N, 1, H, W}, x_conv.options());

    const int threads = 256;
    const int total = N * H * W;
    const int blocks = (total + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    fused_residual_logsumexp_kernel<<<blocks, threads, 0, stream>>>(
        x_conv.data_ptr<float>(),
        x_norm.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W
    );

    return out;
}
"""

_fused_ext = load_inline(
    name="fused_residual_logsumexp_ext",
    cpp_sources=_cpp_source,
    cuda_sources=_cuda_source,
    functions=["fused_residual_logsumexp"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model:
    - Conv2d and GroupNorm remain in PyTorch/cuDNN
    - Tanh + HardSwish + Residual Add + LogSumExp(dim=1, keepdim=True) fused into one custom CUDA op
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self._fused = _fused_ext

    def forward(self, x):
        x_conv = self.conv(x)
        x_norm = self.group_norm(x_conv)
        return self._fused.fused_residual_logsumexp(x_conv.contiguous(), x_norm.contiguous())