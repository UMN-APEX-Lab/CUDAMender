import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void scale_forward_kernel(const float* x, float* y, float scale, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] * scale;
    }
}

__global__ void scale_backward_kernel(const float* grad_out, float* grad_in, float scale, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_in[idx] = grad_out[idx] * scale;
    }
}

torch::Tensor scale_forward_cuda(torch::Tensor x, double scale) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be float32");
    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);
    const int threads = 256;
    const int64_t n = x_contig.numel();
    const int blocks = (int)((n + threads - 1) / threads);
    scale_forward_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<float>(scale),
        n
    );
    return y;
}

torch::Tensor scale_backward_cuda(torch::Tensor grad_out, double scale) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(grad_out.scalar_type() == at::ScalarType::Float, "grad_out must be float32");
    auto go_contig = grad_out.contiguous();
    auto grad_in = torch::empty_like(go_contig);
    const int threads = 256;
    const int64_t n = go_contig.numel();
    const int blocks = (int)((n + threads - 1) / threads);
    scale_backward_kernel<<<blocks, threads>>>(
        go_contig.data_ptr<float>(),
        grad_in.data_ptr<float>(),
        static_cast<float>(scale),
        n
    );
    return grad_in;
}
"""

_cpp_src = r"""
torch::Tensor scale_forward_cuda(torch::Tensor x, double scale);
torch::Tensor scale_backward_cuda(torch::Tensor grad_out, double scale);
"""

_scale_ext = None
if torch.cuda.is_available():
    _scale_ext = load_inline(
        name="scale_cuda_ext_model_opt",
        cpp_sources=_cpp_src,
        cuda_sources=_cuda_src,
        functions=["scale_forward_cuda", "scale_backward_cuda"],
        verbose=False,
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
    )


class _ScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = float(scale)
        if _scale_ext is not None and x.is_cuda and x.dtype == torch.float32:
            return _scale_ext.scale_forward_cuda(x, ctx.scale)
        return x * ctx.scale

    @staticmethod
    def backward(ctx, grad_output):
        if _scale_ext is not None and grad_output.is_cuda and grad_output.dtype == torch.float32:
            grad_input = _scale_ext.scale_backward_cuda(grad_output, ctx.scale)
        else:
            grad_input = grad_output * ctx.scale
        return grad_input, None


class ModelNew(nn.Module):
    """
    Optimized model that keeps Conv2d and BatchNorm2d in PyTorch and replaces
    the final scaling with a custom CUDA operator for FP32 tensors.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = _ScaleFunction.apply(x, self.scaling_factor)
        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 128, 128
kernel_size = 3
scaling_factor = 2.0


def get_inputs():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return [torch.rand(batch_size, in_channels, height, width, device=device)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]