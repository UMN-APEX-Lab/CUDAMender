import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


_cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cfloat>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

__global__ void fused_scale_maxpool_clamp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ scale,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int Hout, int Wout,
    int pool_k,
    float clamp_min,
    float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * Hout * Wout;
    if (idx >= total) return;

    int ow = idx % Wout;
    int oh = (idx / Wout) % Hout;
    int c  = (idx / (Wout * Hout)) % C;
    int n  = idx / (Wout * Hout * C);

    float s = scale[c];
    int hstart = oh * pool_k;
    int wstart = ow * pool_k;

    int base_nc = ((n * C + c) * H) * W;
    float maxv = -FLT_MAX;

    #pragma unroll
    for (int kh = 0; kh < 8; ++kh) {
        if (kh >= pool_k) break;
        int ih = hstart + kh;
        #pragma unroll
        for (int kw = 0; kw < 8; ++kw) {
            if (kw >= pool_k) break;
            int iw = wstart + kw;
            float v = x[base_nc + ih * W + iw] * s;
            maxv = v > maxv ? v : maxv;
        }
    }

    maxv = maxv < clamp_min ? clamp_min : maxv;
    maxv = maxv > clamp_max ? clamp_max : maxv;
    out[idx] = maxv;
}

torch::Tensor fused_scale_maxpool_clamp_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    int64_t pool_k,
    double clamp_min,
    double clamp_max
) {
    CHECK_CUDA(x);
    CHECK_CUDA(scale);
    CHECK_CONTIGUOUS(x);
    CHECK_CONTIGUOUS(scale);
    CHECK_FLOAT(x);
    CHECK_FLOAT(scale);
    TORCH_CHECK(x.dim() == 4, "x must be 4D NCHW");
    TORCH_CHECK(scale.dim() == 3, "scale must be 3D (C,1,1)");
    TORCH_CHECK(scale.size(1) == 1 && scale.size(2) == 1, "scale shape must be (C,1,1)");
    TORCH_CHECK(x.size(1) == scale.size(0), "channel mismatch between x and scale");
    TORCH_CHECK(pool_k > 0, "pool_k must be > 0");
    TORCH_CHECK(x.size(2) % pool_k == 0 && x.size(3) % pool_k == 0, "H and W must be divisible by pool_k");

    auto x_ = x.contiguous();
    auto scale_ = scale.contiguous();

    int N = x_.size(0);
    int C = x_.size(1);
    int H = x_.size(2);
    int W = x_.size(3);
    int Hout = H / pool_k;
    int Wout = W / pool_k;

    auto out = torch::empty({N, C, Hout, Wout}, x_.options());

    int total = N * C * Hout * Wout;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_scale_maxpool_clamp_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x_.data_ptr<float>(),
        scale_.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, Hout, Wout,
        (int)pool_k,
        (float)clamp_min,
        (float)clamp_max
    );

    return out;
}
'''

_cpp_src = r'''
torch::Tensor fused_scale_maxpool_clamp_cuda(
    torch::Tensor x,
    torch::Tensor scale,
    int64_t pool_k,
    double clamp_min,
    double clamp_max
);
'''

_ext = None
if torch.cuda.is_available():
    _ext = load_inline(
        name="fused_scale_maxpool_clamp_ext",
        cpp_sources=_cpp_src,
        cuda_sources=_cuda_src,
        functions=["fused_scale_maxpool_clamp_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


class _FusedScaleMaxPoolClampFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, pool_k, clamp_min, clamp_max):
        ctx.pool_k = int(pool_k)
        ctx.clamp_min = float(clamp_min)
        ctx.clamp_max = float(clamp_max)
        ctx.save_for_backward(x, scale)

        if _ext is not None and x.is_cuda and scale.is_cuda and x.dtype == torch.float32 and scale.dtype == torch.float32:
            return _ext.fused_scale_maxpool_clamp_cuda(
                x.contiguous(), scale.contiguous(), ctx.pool_k, ctx.clamp_min, ctx.clamp_max
            )

        y = x * scale
        y = F.max_pool2d(y, kernel_size=ctx.pool_k)
        y = torch.clamp(y, ctx.clamp_min, ctx.clamp_max)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        pool_k = ctx.pool_k
        clamp_min = ctx.clamp_min
        clamp_max = ctx.clamp_max

        with torch.enable_grad():
            x_ = x.detach().requires_grad_(True)
            scale_ = scale.detach().requires_grad_(True)
            y = x_ * scale_
            y = F.max_pool2d(y, kernel_size=pool_k)
            y = torch.clamp(y, clamp_min, clamp_max)
            grads = torch.autograd.grad(
                y, (x_, scale_), grad_output, retain_graph=False, create_graph=torch.is_grad_enabled()
            )
        return grads[0], grads[1], None, None, None


class ModelNew(nn.Module):
    """
    Optimized model that keeps Conv2d and GroupNorm in PyTorch and fuses
    scale + maxpool + clamp into a custom CUDA operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = self.group_norm(x)
        x = _FusedScaleMaxPoolClampFn.apply(
            x, self.scale, self.maxpool_kernel_size, self.clamp_min, self.clamp_max
        )
        return x