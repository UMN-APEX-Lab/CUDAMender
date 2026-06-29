import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

__global__ void spatial_mean_sub_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int D, int H, int W
) {
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc % C;
    int spatial = D * H * W;
    int base = ((n * C + c) * D) * H * W;

    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < spatial; idx += blockDim.x) {
        sum += x[base + idx];
    }

    __shared__ float sdata[256];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    float mean = sdata[0] / (float)spatial;
    for (int idx = threadIdx.x; idx < spatial; idx += blockDim.x) {
        out[base + idx] = x[base + idx] - mean;
    }
}

torch::Tensor subtract_spatial_mean_cuda(torch::Tensor x) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_FLOAT(x);
    TORCH_CHECK(x.dim() == 5, "x must be a 5D tensor [N, C, D, H, W]");

    const auto N = (int)x.size(0);
    const auto C = (int)x.size(1);
    const auto D = (int)x.size(2);
    const auto H = (int)x.size(3);
    const auto W = (int)x.size(4);

    auto out = torch::empty_like(x);

    const int threads = 256;
    const int blocks = N * C;

    spatial_mean_sub_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W
    );

    return out;
}
'''

_cpp_src = r'''
torch::Tensor subtract_spatial_mean_cuda(torch::Tensor x);
'''

_ext = None
if torch.cuda.is_available():
    _ext = load_inline(
        name='subtract_spatial_mean_ext',
        cpp_sources=_cpp_src,
        cuda_sources=_cuda_src,
        functions=['subtract_spatial_mean_cuda'],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3'],
        verbose=False,
    )


class ModelNew(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self._ext = _ext

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        if x.is_cuda and self._ext is not None:
            x = self._ext.subtract_spatial_mean_cuda(x.contiguous())
        else:
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
        return x


batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]