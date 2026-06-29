import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void min_bias_scale_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    const float constant_value,
    const float scaling_factor,
    const int64_t N,
    const int64_t C,
    const int64_t H,
    const int64_t W
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H * W;
    if (idx < total) {
        int64_t hw = H * W;
        int64_t c = (idx / hw) % C;
        float v = x[idx];
        v = v < constant_value ? v : constant_value;
        v = (v + bias[c]) * scaling_factor;
        out[idx] = v;
    }
}

torch::Tensor min_bias_scale_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    double constant_value,
    double scaling_factor
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW 4D tensor");
    TORCH_CHECK(bias.dim() == 3, "bias must have shape [C,1,1]");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias must have shape [C,1,1]");
    TORCH_CHECK(x.size(1) == bias.size(0), "bias channel dimension must match x channels");

    auto x_contig = x.contiguous();
    auto bias_contig = bias.contiguous();
    auto out = torch::empty_like(x_contig);

    const int64_t N = x_contig.size(0);
    const int64_t C = x_contig.size(1);
    const int64_t H = x_contig.size(2);
    const int64_t W = x_contig.size(3);
    const int64_t total = N * C * H * W;

    const int threads = 256;
    const int blocks = (int)((total + threads - 1) / threads);

    min_bias_scale_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<float>(constant_value),
        static_cast<float>(scaling_factor),
        N, C, H, W
    );

    return out;
}
'''

_cpp_src = r'''
torch::Tensor min_bias_scale_cuda(
    torch::Tensor x,
    torch::Tensor bias,
    double constant_value,
    double scaling_factor
);
'''

_ext = None
if torch.cuda.is_available():
    try:
        _ext = load_inline(
            name=f"min_bias_scale_ext_{os.getpid()}",
            cpp_sources=_cpp_src,
            cuda_sources=_cuda_src,
            functions=["min_bias_scale_cuda"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        _ext = None


class _MinBiasScaleFn:
    @staticmethod
    def apply(x, bias, constant_value, scaling_factor):
        if (
            _ext is not None
            and x.is_cuda
            and bias.is_cuda
            and x.dtype == torch.float32
            and bias.dtype == torch.float32
        ):
            return _ext.min_bias_scale_cuda(x, bias, float(constant_value), float(scaling_factor))
        return (torch.minimum(x, torch.tensor(float(constant_value), device=x.device, dtype=x.dtype)) + bias) * float(scaling_factor)


class ModelNew(nn.Module):
    """
    Optimized model that performs convolution followed by a fused CUDA kernel for:
    min with constant -> add bias -> scale
    """
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = float(constant_value)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv(x)
        x = _MinBiasScaleFn.apply(x, self.bias, self.constant_value, self.scaling_factor)
        return x


batch_size = 128
in_channels = 64
out_channels = 128
height = width = 128
kernel_size = 3
constant_value = 0.5
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor]