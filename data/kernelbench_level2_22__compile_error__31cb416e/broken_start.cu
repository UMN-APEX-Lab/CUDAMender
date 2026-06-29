import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

__device__ __forceinline__ float softplus_stable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__global__ void fused_post_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int rows,
    int cols,
    float mult,
    float clamp_min_v,
    float clamp_max_v
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    extern __shared__ float sdata[];
    float* smax = sdata;
    float* ssum = sdata + blockDim.x;

    const float* row_ptr = in + (size_t)row * cols;

    float local_max = -CUDART_INF_F;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = row_ptr[col] * mult;
        v = fminf(fmaxf(v, clamp_min_v), clamp_max_v);
        local_max = fmaxf(local_max, v);
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smax[tid] = fmaxf(smax[tid], smax[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = smax[0];
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = row_ptr[col] * mult;
        v = fminf(fmaxf(v, clamp_min_v), clamp_max_v);
        local_sum += expf(v - row_max);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float lse = row_max + logf(ssum[0]);
        float mish = lse * tanhf(softplus_stable(lse));
        out[row] = lse * mish;
    }
}

torch::Tensor fused_post_cuda(
    torch::Tensor input,
    double mult,
    double clamp_min_v,
    double clamp_max_v
) {
    CHECK_CUDA(input);
    CHECK_CONTIGUOUS(input);
    CHECK_FLOAT(input);
    TORCH_CHECK(input.dim() == 2, "input must be 2D");

    auto rows = (int)input.size(0);
    auto cols = (int)input.size(1);
    auto output = torch::empty({rows, 1}, input.options());

    const int threads = 256;
    const int blocks = rows;
    const size_t shmem = 2 * threads * sizeof(float);

    fused_post_kernel<<<blocks, threads, shmem>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        (float)mult,
        (float)clamp_min_v,
        (float)clamp_max_v
    );

    return output;
}
"""

_cpp_src = r"""
torch::Tensor fused_post_cuda(
    torch::Tensor input,
    double mult,
    double clamp_min_v,
    double clamp_max_v
);
"""

_fused_mod = None
if torch.cuda.is_available():
    try:
        _fused_mod = load_inline(
            name="fused_post_ext_v1",
            cpp_sources=_cpp_src,
            cuda_sources=_cuda_src,
            functions=["fused_post_cuda"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception:
        _fused_mod = None


class ModelNew(nn.Module):
    """
    Optimized model:
    - keeps matmul as nn.Linear (cuBLAS/cuDNN optimized)
    - fuses scale, residual add, clamp, logsumexp reduction, and final x*mish(x) into one CUDA kernel
    """
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self._fused = _fused_mod
        self._mult = 2.0 * self.scale_factor

    def forward(self, x):
        x = self.matmul(x)
        if (
            self._fused is not None
            and x.is_cuda
            and x.dtype == torch.float32
            and x.dim() == 2
        ):
            return self._fused.fused_post_cuda(x.contiguous(), self._mult, self.clamp_min, self.clamp_max)

        x = x * self.scale_factor
        x = x + x
        x = torch.clamp(x, self.clamp_min, self.clamp_max)
        x = torch.logsumexp(x, dim=1, keepdim=True)
        x = x * F.mish(x)
        return x