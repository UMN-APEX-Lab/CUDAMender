import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_cpp_src = r"""
torch::Tensor mish_tanh_cuda(torch::Tensor x);
"""

_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float softplus_approx(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ __forceinline__ float mish_tanh_op(float x) {
    float sp = softplus_approx(x);
    float mish = x * tanhf(sp);
    return tanhf(mish);
}

__global__ void mish_tanh_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = mish_tanh_op(x[idx]);
    }
}

torch::Tensor mish_tanh_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "mish_tanh_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::kFloat, "mish_tanh_cuda: only float32 is supported");
    auto x_contig = x.contiguous();
    auto out = torch::empty_like(x_contig);

    const int threads = 256;
    const int64_t n = x_contig.numel();
    const int blocks = (int)((n + threads - 1) / threads);

    mish_tanh_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}
"""

try:
    _mish_tanh_ext = load_inline(
        name="mish_tanh_ext_v1",
        cpp_sources=_cpp_src,
        cuda_sources=_cuda_src,
        functions=["mish_tanh_cuda"],
        verbose=False,
    )
except Exception:
    _mish_tanh_ext = None


class ModelNew(nn.Module):
    """
    Optimized model that uses cuDNN Conv3d and a fused custom CUDA kernel for Mish followed by Tanh.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        if x.is_cuda and x.dtype == torch.float32 and _mish_tanh_ext is not None:
            return _mish_tanh_ext.mish_tanh_cuda(x)
        return torch.tanh(F.mish(x))


batch_size = 16
in_channels = 32
out_channels = 64
D, H, W = 32, 64, 64
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, D, H, W)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]