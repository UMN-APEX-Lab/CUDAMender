import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_ext = None

if torch.cuda.is_available():
    _cpp_source = r"""
    torch::Tensor exclusive_cumsum_cuda(torch::Tensor x);
    """

    _cuda_source = r"""
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <vector>

    __global__ void exclusive_cumsum_rowwise_kernel(
        const float* __restrict__ x,
        float* __restrict__ out,
        const int rows,
        const int cols
    ) {
        int row = blockIdx.x;
        if (row >= rows) return;

        const float* x_row = x + ((long long)row) * cols;
        float* out_row = out + ((long long)row) * cols;

        float acc = 0.0f;
        out_row[0] = 0.0f;
        for (int col = 1; col < cols; ++col) {
            acc += x_row[col - 1];
            out_row[col] = acc;
        }
    }

    torch::Tensor exclusive_cumsum_cuda(torch::Tensor x) {
        TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
        TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
        TORCH_CHECK(x.dim() == 2, "x must be 2D");
        TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

        const auto rows = static_cast<int>(x.size(0));
        const auto cols = static_cast<int>(x.size(1));

        auto out = torch::empty_like(x);

        dim3 grid(rows);
        dim3 block(1);

        exclusive_cumsum_rowwise_kernel<<<grid, block>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            rows,
            cols
        );

        return out;
    }
    """

    try:
        _ext = load_inline(
            name="exclusive_cumsum_ext_v1",
            cpp_sources=_cpp_source,
            cuda_sources=_cuda_source,
            functions=["exclusive_cumsum_cuda"],
            verbose=False,
        )
    except Exception:
        _ext = None


class ModelNew(nn.Module):
    """
    Optimized model for exclusive cumulative sum along dim=1 on CUDA FP32 tensors.
    Falls back to the original PyTorch implementation for unsupported cases.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self._ext = _ext

    def forward(self, x):
        if (
            self._ext is not None
            and x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 2
            and self.dim == 1
        ):
            if not x.is_contiguous():
                x = x.contiguous()
            return self._ext.exclusive_cumsum_cuda(x)

        cumsum = torch.cumsum(
            x.narrow(dim=self.dim, start=0, length=x.size(self.dim) - 1),
            dim=self.dim,
        )
        return torch.cat(
            (torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), cumsum),
            dim=self.dim,
        )