import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

__global__ void gelu_global_avg_pool2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W
) {
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc % C;
    int tid = threadIdx.x;
    int HW = H * W;

    const float* base = x + ((n * C + c) * H * W);

    float sum = 0.0f;
    for (int idx = tid; idx < HW; idx += blockDim.x) {
        float v = base[idx];
        sum += gelu_exact(v);
    }

    __shared__ float sdata[256];
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[n * C + c] = sdata[0] / static_cast<float>(HW);
    }
}

torch::Tensor gelu_global_avg_pool2d_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be a 4D tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    const auto N = static_cast<int>(x.size(0));
    const auto C = static_cast<int>(x.size(1));
    const auto H = static_cast<int>(x.size(2));
    const auto W = static_cast<int>(x.size(3));

    auto out = torch::empty({N, C}, x.options());

    constexpr int threads = 256;
    const int blocks = N * C;

    auto stream = at::cuda::getDefaultCUDAStream();
    gelu_global_avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W
    );

    return out;
}
"""

_cpp_src = r"""
torch::Tensor gelu_global_avg_pool2d_cuda(torch::Tensor x);
"""

_ext = load_inline(
    name="gelu_global_avg_pool2d_ext",
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=["gelu_global_avg_pool2d_cuda"],
    extra_cuda_cflags=["-O3"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Optimized model that keeps cuDNN convolution and fuses GELU + global average pooling
    into a custom CUDA operator for FP32 tensors.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        if x.is_cuda and x.dtype == torch.float32:
            return _ext.gelu_global_avg_pool2d_cuda(x.contiguous())
        x = F.gelu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x


batch_size = 128
in_channels = 8
out_channels = 64
height, width = 256, 256
kernel_size = 3


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]