import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cpp_source = r"""
torch::Tensor rowwise_scaled_dot_cuda(torch::Tensor x, torch::Tensor v, double scale);
"""

_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void rowwise_scaled_dot_kernel(
    const float* __restrict__ x,
    const float* __restrict__ v,
    float* __restrict__ out,
    int B,
    int N,
    float scale
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];
    float sum = 0.0f;

    const float* x_row = x + (size_t)row * N;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += x_row[i] * v[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[row] = sdata[0] * scale;
    }
}

torch::Tensor rowwise_scaled_dot_cuda(torch::Tensor x, torch::Tensor v, double scale) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(v.dim() == 1, "v must be 1D");
    TORCH_CHECK(x.size(1) == v.size(0), "x.size(1) must equal v.size(0)");

    auto x_contig = x.contiguous();
    auto v_contig = v.contiguous();

    const int B = (int)x_contig.size(0);
    const int N = (int)x_contig.size(1);

    auto out = torch::empty({B, 1}, x_contig.options());

    const int threads = 256;
    const dim3 blocks(B);
    const size_t shmem = threads * sizeof(float);

    rowwise_scaled_dot_kernel<<<blocks, threads, shmem>>>(
        x_contig.data_ptr<float>(),
        v_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        B,
        N,
        static_cast<float>(scale)
    );

    return out;
}
"""

_rowwise_scaled_dot = load_inline(
    name="rowwise_scaled_dot_ext",
    cpp_sources=_cpp_source,
    cuda_sources=_cuda_source,
    functions=["rowwise_scaled_dot_cuda"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model using the identity:
    sum_j((x @ W^T)[:, j]) = x @ sum_j(W[j, :])
    so the original computation becomes a row-wise dot product with the
    pre-aggregated weight vector, fused with the final scaling and division.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self._rowwise_scaled_dot = _rowwise_scaled_dot

    def forward(self, x):
        scale = float(self.scaling_factor) * 0.5
        weight_sum = self.weight.sum(dim=0)
        if x.is_cuda and self.weight.is_cuda and x.dtype == torch.float32 and self.weight.dtype == torch.float32:
            return self._rowwise_scaled_dot.rowwise_scaled_dot_cuda(x, weight_sum, scale)
        return (x.matmul(weight_sum.unsqueeze(1))) * scale