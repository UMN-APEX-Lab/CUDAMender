import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_mish_ext = None


def _load_mish_extension():
    global _mish_ext
    if _mish_ext is not None:
        return _mish_ext

    if not torch.cuda.is_available():
        return None

    cpp_source = r"""
    #include <torch/extension.h>
    torch::Tensor mish_mul_cuda(torch::Tensor x);
    """

    cuda_source = r"""
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <cuda.h>
    #include <cuda_runtime.h>

    __device__ __forceinline__ float softplus_stable(float x) {
        if (x > 20.0f) return x;
        if (x < -20.0f) return expf(x);
        return log1pf(expf(x));
    }

    __global__ void mish_mul_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
        int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float v = x[idx];
            float sp = softplus_stable(v);
            out[idx] = v * tanhf(sp);
        }
    }

    torch::Tensor mish_mul_cuda(torch::Tensor x) {
        TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
        TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be float32");
        auto x_contig = x.contiguous();
        auto out = torch::empty_like(x_contig);
        int64_t n = x_contig.numel();

        const int threads = 256;
        const int blocks = (int)((n + threads - 1) / threads);

        mish_mul_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            x_contig.data_ptr<float>(),
            out.data_ptr<float>(),
            n
        );

        return out;
    }
    """

    try:
        _mish_ext = load_inline(
            name="mish_mul_ext_v1",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["mish_mul_cuda"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:
        _mish_ext = None

    return _mish_ext


class _MishMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ext = _load_mish_extension()
        if x.is_cuda and ext is not None and x.dtype == torch.float32:
            return ext.mish_mul_cuda(x)
        return x * torch.tanh(F.softplus(x))

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        sp = F.softplus(x)
        tsp = torch.tanh(sp)
        sig = torch.sigmoid(x)
        grad = tsp + x * sig * (1 - tsp * tsp)
        return grad_output * grad


class ModelNew(nn.Module):
    """
    Optimized model with custom CUDA fused activation: x * tanh(softplus(x)).
    Convolution and BatchNorm use PyTorch implementations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.conv(x)
        x = _MishMulFunction.apply(x)
        x = self.bn(x)
        return x


batch_size = 64
in_channels = 64
out_channels = 128
height, width = 128, 128
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]