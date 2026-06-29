import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


_cpp_source = r"""
torch::Tensor fused_norm_add_mul_cuda(torch::Tensor x, torch::Tensor y, double eps);
"""

_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

__global__ void fused_norm_add_mul_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int rows,
    int cols,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* x_row = x + (size_t)row * cols;
    const float* y_row = y + (size_t)row * cols;
    float* out_row = out + (size_t)row * cols;

    __shared__ float s_sum[256];
    __shared__ float s_sqsum[256];

    float local_sum = 0.0f;
    float local_sqsum = 0.0f;

    for (int col = tid; col < cols; col += blockDim.x) {
        float v = x_row[col];
        local_sum += v;
        local_sqsum += v * v;
    }

    s_sum[tid] = local_sum;
    s_sqsum[tid] = local_sqsum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sqsum[tid] += s_sqsum[tid + stride];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)cols;
    float var = s_sqsum[0] / (float)cols - mean * mean;
    float inv_std = rsqrtf(var > 0.0f ? (var + eps) : eps);

    for (int col = tid; col < cols; col += blockDim.x) {
        float xv = x_row[col];
        float yv = y_row[col];
        float norm = (xv - mean) * inv_std;
        out_row[col] = (norm + yv) * yv;
    }
}

torch::Tensor fused_norm_add_mul_cuda(torch::Tensor x, torch::Tensor y, double eps) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(y.dim() == 2, "y must be 2D");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto rows = (int)x.size(0);
    auto cols = (int)x.size(1);

    auto out = torch::empty_like(x);

    const int threads = 256;
    const dim3 blocks(rows);
    auto stream = at::cuda::getDefaultCUDAStream();

    fused_norm_add_mul_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols,
        static_cast<float>(eps)
    );

    return out;
}
"""

_fused_ops = load_inline(
    name="fused_norm_add_mul_ext",
    cpp_sources=_cpp_source,
    cuda_sources=_cuda_source,
    functions=["fused_norm_add_mul_cuda"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that uses a fused CUDA kernel for row-wise instance normalization,
    residual addition, and multiplication.
    """
    def __init__(self, in_features, out_features, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.bmm = nn.Linear(in_features, out_features)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x, y):
        x = self.bmm(x)
        if x.is_cuda and y.is_cuda and x.dtype == torch.float32 and y.dtype == torch.float32:
            return _fused_ops.fused_norm_add_mul_cuda(x.contiguous(), y.contiguous(), float(self.eps))
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x + y
        x = x * y
        return x