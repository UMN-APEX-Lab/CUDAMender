import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_cpp_source = r"""
torch::Tensor avg_pool2d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding);
"""

_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void avg_pool2d_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c  = (idx / (outW * outH)) % C;
    int n  = idx / (outW * outH * C);

    int hstart = oh * stride - padding;
    int wstart = ow * stride - padding;

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = hstart + kh;
        if ((unsigned)ih >= (unsigned)H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int iw = wstart + kw;
            if ((unsigned)iw >= (unsigned)W) continue;
            int in_idx = ((n * C + c) * H + ih) * W + iw;
            sum += x[in_idx];
        }
    }

    out[idx] = sum / static_cast<float>(kernel_size * kernel_size);
}

torch::Tensor avg_pool2d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "avg_pool2d_cuda: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "avg_pool2d_cuda: only float32 is supported");
    TORCH_CHECK(x.dim() == 4, "avg_pool2d_cuda: input must be 4D NCHW");

    auto x_contig = x.contiguous();

    const int64_t N = x_contig.size(0);
    const int64_t C = x_contig.size(1);
    const int64_t H = x_contig.size(2);
    const int64_t W = x_contig.size(3);

    TORCH_CHECK(kernel_size > 0, "kernel_size must be > 0");
    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    const int64_t outH = (H + 2 * padding - kernel_size) / stride + 1;
    const int64_t outW = (W + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(outH >= 0 && outW >= 0, "Calculated output size is too small");

    auto out = torch::empty({N, C, outH, outW}, x_contig.options());

    const int threads = 256;
    const int64_t total = N * C * outH * outW;
    const int blocks = static_cast<int>((total + threads - 1) / threads);

    avg_pool2d_forward_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(outH),
        static_cast<int>(outW),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding)
    );

    return out;
}
"""

_avg_pool2d_ext = None
if torch.cuda.is_available():
    _avg_pool2d_ext = load_inline(
        name="avg_pool2d_cuda_ext_opt",
        cpp_sources=_cpp_source,
        cuda_sources=_cuda_source,
        functions=["avg_pool2d_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using a custom CUDA operator
    for FP32 CUDA tensors, with a PyTorch fallback otherwise.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(kernel_size if stride is None else stride)
        self.padding = int(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            _avg_pool2d_ext is not None
            and x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 4
        ):
            return _avg_pool2d_ext.avg_pool2d_cuda(
                x, self.kernel_size, self.stride, self.padding
            )
        return F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )