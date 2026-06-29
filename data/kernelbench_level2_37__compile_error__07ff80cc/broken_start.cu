import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cpp_source = r"""
torch::Tensor swish_bias_cuda(torch::Tensor x, torch::Tensor bias);
"""

_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void swish_bias_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) {
        int c = idx % cols;
        float v = x[idx];
        float s = 1.0f / (1.0f + expf(-v));
        out[idx] = v * s + bias[c];
    }
}

torch::Tensor swish_bias_cuda(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(x.size(1) == bias.size(0), "bias shape mismatch");

    auto x_contig = x.contiguous();
    auto bias_contig = bias.contiguous();
    auto out = torch::empty_like(x_contig);

    int rows = (int)x_contig.size(0);
    int cols = (int)x_contig.size(1);
    int total = rows * cols;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    swish_bias_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}
"""

_swish_bias_ext = load_inline(
    name="swish_bias_ext_v1",
    cpp_sources=_cpp_source,
    cuda_sources=_cuda_source,
    functions=["swish_bias_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)

class _SwishBiasFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias):
        if x.is_cuda and bias.is_cuda and x.dtype == torch.float32 and bias.dtype == torch.float32:
            ctx.save_for_backward(x, bias)
            return _swish_bias_ext.swish_bias_cuda(x, bias)
        ctx.save_for_backward(x, bias)
        return x * torch.sigmoid(x) + bias

    @staticmethod
    def backward(ctx, grad_output):
        x, bias = ctx.saved_tensors
        sig = torch.sigmoid(x)
        grad_x = grad_output * (sig + x * sig * (1 - sig))
        reduce_dims = tuple(range(grad_output.dim() - bias.dim()))
        grad_bias = grad_output.sum(dim=reduce_dims) if reduce_dims else grad_output
        return grad_x, grad_bias

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

    def forward(self, x):
        x = self.matmul(x)
        x = _SwishBiasFunction.apply(x, self.bias)
        x = self.group_norm(x)
        return x