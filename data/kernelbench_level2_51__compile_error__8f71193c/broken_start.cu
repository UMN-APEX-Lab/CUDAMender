import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_cpp_source = r"""
torch::Tensor fused_post_cuda(torch::Tensor linear_out, torch::Tensor subtract, torch::Tensor original_x);
"""

_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ float gelu_approx(float x) {
    const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
    const float kBeta = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
}

template <int THREADS>
__global__ void fused_post_kernel(
    const float* __restrict__ linear_out,   // [B, O]
    const float* __restrict__ subtract,     // [O]
    const float* __restrict__ original_x,   // [B, I]
    float* __restrict__ out,                // [B, I]
    int B, int O, int I
) {
    int row = blockIdx.x;
    if (row >= B) return;

    __shared__ float sdata[THREADS];
    float sum = 0.0f;

    const float* row_linear = linear_out + (size_t)row * O;
    const float* row_orig = original_x + (size_t)row * I;
    float* row_out = out + (size_t)row * I;

    for (int col = threadIdx.x; col < O; col += THREADS) {
        sum += row_linear[col] - subtract[col];
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float g = gelu_approx(sdata[0] / (float)O);

    for (int col = threadIdx.x; col < I; col += THREADS) {
        row_out[col] = row_orig[col] + g;
    }
}

torch::Tensor fused_post_cuda(torch::Tensor linear_out, torch::Tensor subtract, torch::Tensor original_x) {
    TORCH_CHECK(linear_out.is_cuda(), "linear_out must be CUDA");
    TORCH_CHECK(subtract.is_cuda(), "subtract must be CUDA");
    TORCH_CHECK(original_x.is_cuda(), "original_x must be CUDA");
    TORCH_CHECK(linear_out.dtype() == torch::kFloat32, "linear_out must be float32");
    TORCH_CHECK(subtract.dtype() == torch::kFloat32, "subtract must be float32");
    TORCH_CHECK(original_x.dtype() == torch::kFloat32, "original_x must be float32");
    TORCH_CHECK(linear_out.dim() == 2, "linear_out must be 2D");
    TORCH_CHECK(subtract.dim() == 1, "subtract must be 1D");
    TORCH_CHECK(original_x.dim() == 2, "original_x must be 2D");
    TORCH_CHECK(linear_out.size(0) == original_x.size(0), "batch size mismatch");
    TORCH_CHECK(linear_out.size(1) == subtract.size(0), "out_features mismatch with subtract");

    auto linear_c = linear_out.contiguous();
    auto subtract_c = subtract.contiguous();
    auto original_c = original_x.contiguous();

    const int B = (int)linear_c.size(0);
    const int O = (int)linear_c.size(1);
    const int I = (int)original_c.size(1);

    auto out = torch::empty_like(original_c);

    constexpr int THREADS = 256;
    dim3 blocks(B);
    dim3 threads(THREADS);

    fused_post_kernel<THREADS><<<blocks, threads>>>(
        linear_c.data_ptr<float>(),
        subtract_c.data_ptr<float>(),
        original_c.data_ptr<float>(),
        out.data_ptr<float>(),
        B, O, I
    );

    return out;
}
"""

_fused_mod = None
if torch.cuda.is_available():
    _fused_mod = load_inline(
        name="fused_gemm_post_ops_ext",
        cpp_sources=_cpp_source,
        cuda_sources=_cuda_source,
        functions=["fused_post_cuda"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


class ModelNew(nn.Module):
    """
    Optimized model:
    - Uses cuBLAS-backed F.linear for GEMM
    - Fuses Subtract + GlobalAvgPool + LogSumExp(singleton simplification) + GELU + ResidualAdd
      into one custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.detach()
        x_lin = F.linear(x, self.gemm.weight, self.gemm.bias)

        if (
            _fused_mod is not None
            and x_lin.is_cuda
            and original_x.is_cuda
            and self.subtract.is_cuda
            and x_lin.dtype == torch.float32
            and original_x.dtype == torch.float32
            and self.subtract.dtype == torch.float32
        ):
            return _fused_mod.fused_post_cuda(x_lin, self.subtract, original_x)

        x = x_lin - self.subtract
        x = torch.mean(x, dim=1, keepdim=True)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = F.gelu(x)
        x = x + original_x
        return x