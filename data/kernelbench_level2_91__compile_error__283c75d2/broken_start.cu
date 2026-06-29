import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>

__global__ void fused_softmax_bias_scale_sigmoid_kernel(
    const float* __restrict__ x,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C, int H, int W,
    float scaling_factor
) {
    int nhw = blockIdx.x;
    int tid = threadIdx.x;
    int HW = H * W;
    int total_positions = N * HW;
    if (nhw >= total_positions) return;

    int n = nhw / HW;
    int hw = nhw % HW;
    int base = n * C * HW + hw;

    __shared__ float smax[256];
    __shared__ float ssum[256];

    float local_max = -FLT_MAX;
    for (int c = tid; c < C; c += blockDim.x) {
        float v = x[base + c * HW];
        local_max = v > local_max ? v : local_max;
    }
    smax[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = smax[tid + stride];
            smax[tid] = smax[tid] > other ? smax[tid] : other;
        }
        __syncthreads();
    }
    float max_val = smax[0];

    float local_sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        local_sum += expf(x[base + c * HW] - max_val);
    }
    ssum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = ssum[0];

    for (int c = tid; c < C; c += blockDim.x) {
        float soft = expf(x[base + c * HW] - max_val) / sum_val;
        float z = (soft + bias[c]) * scaling_factor;
        out[base + c * HW] = 1.0f / (1.0f + expf(-z));
    }
}

torch::Tensor fused_post_cuda(torch::Tensor x, torch::Tensor bias, double scaling_factor) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be NCHW");
    TORCH_CHECK(bias.numel() == x.size(1), "bias must have numel equal to channel dimension");

    auto x_contig = x.contiguous();
    auto bias_flat = bias.contiguous().view({-1});
    auto out = torch::empty_like(x_contig);

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    int threads = 256;
    int blocks = N * H * W;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    fused_softmax_bias_scale_sigmoid_kernel<<<blocks, threads, 0, stream>>>(
        x_contig.data_ptr<float>(),
        bias_flat.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        static_cast<float>(scaling_factor)
    );

    return out;
}
'''

_cpp_src = r'''
torch::Tensor fused_post_cuda(torch::Tensor x, torch::Tensor bias, double scaling_factor);
'''

_fused_module = None
if torch.cuda.is_available():
    _fused_module = load_inline(
        name='fused_softmax_bias_scale_sigmoid_ext',
        cpp_sources=_cpp_src,
        cuda_sources=_cuda_src,
        functions=['fused_post_cuda'],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--use_fast_math'],
        verbose=False,
    )


class ModelNew(nn.Module):
    """
    Optimized model: keeps ConvTranspose2d in PyTorch and fuses softmax(channel-wise) + bias add + scale + sigmoid into one CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))
        self.scaling_factor = float(scaling_factor)

    def forward(self, x):
        x = self.conv_transpose(x)
        if x.is_cuda and _fused_module is not None and x.dtype == torch.float32 and self.bias.dtype == torch.float32:
            return _fused_module.fused_post_cuda(x, self.bias, self.scaling_factor)
        x = torch.softmax(x, dim=1)
        x = x + self.bias
        x = x * self.scaling_factor
        x = torch.sigmoid(x)
        return x


batch_size = 128
in_channels = 64
out_channels = 128
height, width = 64, 64
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0


def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width, dtype=torch.float32).cuda()]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]