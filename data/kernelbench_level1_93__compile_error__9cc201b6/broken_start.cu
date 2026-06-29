import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void masked_cumsum_lastdim_kernel(
    const float* __restrict__ x,
    const bool* __restrict__ mask,
    float* __restrict__ out,
    int64_t rows,
    int64_t cols
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t row = tid; row < rows; row += stride) {
        int64_t base = row * cols;
        float acc = 0.0f;
        for (int64_t col = 0; col < cols; ++col) {
            float v = mask[base + col] ? x[base + col] : 0.0f;
            acc += v;
            out[base + col] = acc;
        }
    }
}

torch::Tensor masked_cumsum_lastdim_cuda(torch::Tensor x, torch::Tensor mask) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "mask must be bool");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mask.is_contiguous(), "mask must be contiguous");
    TORCH_CHECK(x.sizes() == mask.sizes(), "x and mask must have the same shape");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");

    auto out = torch::empty_like(x);

    int64_t cols = x.size(-1);
    int64_t rows = x.numel() / cols;

    const int threads = 256;
    int blocks = (int)((rows + threads - 1) / threads);
    blocks = blocks > 4096 ? 4096 : blocks;
    if (blocks < 1) blocks = 1;

    masked_cumsum_lastdim_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        mask.data_ptr<bool>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}
"""

_cpp_src = r"""
torch::Tensor masked_cumsum_lastdim_cuda(torch::Tensor x, torch::Tensor mask);
"""

_ext = None
if torch.cuda.is_available():
    _ext = load_inline(
        name="masked_cumsum_ext_v1",
        cpp_sources=_cpp_src,
        cuda_sources=_cuda_src,
        functions=["masked_cumsum_lastdim_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x, mask):
        if (
            _ext is None
            or not x.is_cuda
            or not mask.is_cuda
            or x.dtype != torch.float32
            or mask.dtype != torch.bool
            or x.shape != mask.shape
        ):
            return torch.cumsum(x * mask, dim=self.dim)

        dim = self.dim if self.dim >= 0 else self.dim + x.dim()
        if dim < 0 or dim >= x.dim():
            return torch.cumsum(x * mask, dim=self.dim)

        if dim != x.dim() - 1:
            perm = [i for i in range(x.dim()) if i != dim] + [dim]
            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            x_t = x.permute(perm).contiguous()
            m_t = mask.permute(perm).contiguous()
            y_t = _ext.masked_cumsum_lastdim_cuda(x_t, m_t)
            return y_t.permute(inv_perm)
        else:
            return _ext.masked_cumsum_lastdim_cuda(x.contiguous(), mask.contiguous())