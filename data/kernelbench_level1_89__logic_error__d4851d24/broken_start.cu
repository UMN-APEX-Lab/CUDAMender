import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void cumsum_lastdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t rows,
    int64_t cols
) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows) return;

    extern __shared__ float shmem[];
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int64_t base = row * cols;
    float carry = 0.0f;

    for (int64_t start = 0; start < cols; start += block_size) {
        int64_t idx = start + tid;
        int tile_elems = (int)((cols - start) > block_size ? block_size : (cols - start));

        float v = 0.0f;
        if (tid < tile_elems) {
            v = x[base + idx];
        }
        shmem[tid] = v;
        __syncthreads();

        for (int offset = 1; offset < block_size; offset <<= 1) {
            float addv = 0.0f;
            if (tid >= offset) {
                addv = shmem[tid - offset];
            }
            __syncthreads();
            if (tid < tile_elems) {
                shmem[tid] += addv;
            }
            __syncthreads();
        }

        if (tid < tile_elems) {
            out[base + idx] = shmem[tid] + carry;
        }
        __syncthreads();

        if (tid == 0) {
            carry += shmem[tile_elems - 1];
        }
        __syncthreads();
    }
}

torch::Tensor cumsum_lastdim_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");

    auto out = torch::empty_like(x);
    const int64_t cols = x.size(-1);
    const int64_t rows = x.numel() / cols;

    if (x.numel() == 0) {
        return out;
    }

    const int threads = 256;
    const dim3 blocks((unsigned int)rows);
    const size_t shmem = threads * sizeof(float);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    cumsum_lastdim_kernel<<<blocks, threads, shmem, stream>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        rows,
        cols
    );

    return out;
}
'''

_cpp_src = r'''
torch::Tensor cumsum_lastdim_cuda(torch::Tensor x);
'''

_cumsum_ext = load_inline(
    name='cumsum_lastdim_ext_v1',
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=['cumsum_lastdim_cuda'],
    extra_cuda_cflags=['-O3'],
    extra_cflags=['-O3'],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self._ext = _cumsum_ext

    def forward(self, x):
        dim = self.dim if self.dim >= 0 else x.dim() + self.dim
        if x.is_cuda and x.dtype == torch.float32 and x.dim() >= 1 and dim == x.dim() - 1:
            xc = x.contiguous()
            return self._ext.cumsum_lastdim_cuda(xc)
        return torch.cumsum(x, dim=self.dim)