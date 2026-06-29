import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_avgpool2d_cpp_source = r"""
torch::Tensor avg_pool2d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding);
"""

_avgpool2d_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

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
    int hend = hstart + kernel_size;
    int wend = wstart + kernel_size;

    float sum = 0.0f;

    int base_nc = ((n * C + c) * H) * W;
    for (int ih = hstart; ih < hend; ++ih) {
        if ((unsigned)ih < (unsigned)H) {
            int row_base = base_nc + ih * W;
            for (int iw = wstart; iw < wend; ++iw) {
                if ((unsigned)iw < (unsigned)W) {
                    sum += x[row_base + iw];
                }
            }
        }
    }

    out[idx] = sum / float(kernel_size * kernel_size);
}

torch::Tensor avg_pool2d_cuda(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor of shape (N, C, H, W)");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be > 0");
    TORCH_CHECK(stride > 0, "stride must be > 0");
    TORCH_CHECK(padding >= 0, "padding must be >= 0");

    const auto N = (int)x.size(0);
    const auto C = (int)x.size(1);
    const auto H = (int)x.size(2);
    const auto W = (int)x.size(3);

    const int outH = (H + 2 * (int)padding - (int)kernel_size) / (int)stride + 1;
    const int outW = (W + 2 * (int)padding - (int)kernel_size) / (int)stride + 1;

    TORCH_CHECK(outH >= 0 && outW >= 0, "Calculated output size is too small");

    auto out = torch::empty({N, C, outH, outW}, x.options());

    const int threads = 256;
    const int total = N * C * outH * outW;
    const int blocks = (total + threads - 1) / threads;

    avg_pool2d_forward_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W, outH, outW,
        (int)kernel_size, (int)stride, (int)padding
    );

    return out;
}
"""

_avgpool2d_ext = None


def _get_avgpool2d_ext():
    global _avgpool2d_ext
    if _avgpool2d_ext is None:
        _avgpool2d_ext = load_inline(
            name="avgpool2d_cuda_ext_v1",
            cpp_sources=_avgpool2d_cpp_source,
            cuda_sources=_avgpool2d_cuda_source,
            functions=["avg_pool2d_cuda"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _avgpool2d_ext


class ModelNew(nn.Module):
    """
    Optimized model that performs 2D Average Pooling using a custom CUDA kernel for FP32 inputs.
    Falls back to PyTorch AvgPool2d on CPU/non-FP32 tensors.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super(ModelNew, self).__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(kernel_size if stride is None else stride)
        self.padding = int(padding)
        self._fallback = nn.AvgPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.dtype == torch.float32:
            ext = _get_avgpool2d_ext()
            return ext.avg_pool2d_cuda(x.contiguous(), self.kernel_size, self.stride, self.padding)
        return self._fallback(x)