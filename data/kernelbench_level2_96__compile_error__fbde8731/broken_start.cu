import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cuda_src = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)

__global__ void fused_post_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int D, int H, int W,
    int pool_k,
    float scale,
    float clamp_min,
    float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C;
    if (idx >= total) return;

    int n = idx / C;
    int c = idx % C;

    int outD = D / pool_k;
    int outH = H / pool_k;
    int outW = W / pool_k;

    const int base_nc = ((n * C + c) * D) * H * W;

    float sum = 0.0f;
    int pooled_count = outD * outH * outW;

    for (int od = 0; od < outD; ++od) {
        int id0 = od * pool_k;
        for (int oh = 0; oh < outH; ++oh) {
            int ih0 = oh * pool_k;
            for (int ow = 0; ow < outW; ++ow) {
                int iw0 = ow * pool_k;

                float m = -FLT_MAX;
                for (int kd = 0; kd < pool_k; ++kd) {
                    int id = id0 + kd;
                    for (int kh = 0; kh < pool_k; ++kh) {
                        int ih = ih0 + kh;
                        for (int kw = 0; kw < pool_k; ++kw) {
                            int iw = iw0 + kw;
                            int offset = base_nc + (id * H + ih) * W + iw;
                            float v = x[offset] * scale;
                            m = v > m ? v : m;
                        }
                    }
                }
                sum += m;
            }
        }
    }

    float avg = pooled_count > 0 ? sum / (float)pooled_count : 0.0f;
    avg = avg < clamp_min ? clamp_min : avg;
    avg = avg > clamp_max ? clamp_max : avg;

    out[idx] = avg;
}

torch::Tensor fused_post_cuda(
    torch::Tensor x,
    int64_t pool_k,
    double scale,
    double clamp_min,
    double clamp_max
) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 5, "x must be a 5D tensor [N, C, D, H, W]");
    TORCH_CHECK(pool_k > 0, "pool_k must be > 0");

    const auto N = (int)x.size(0);
    const auto C = (int)x.size(1);
    const auto D = (int)x.size(2);
    const auto H = (int)x.size(3);
    const auto W = (int)x.size(4);

    TORCH_CHECK(D >= pool_k && H >= pool_k && W >= pool_k, "pool kernel too large for input");
    TORCH_CHECK(D % pool_k == 0 && H % pool_k == 0 && W % pool_k == 0,
                "This fused kernel currently requires D/H/W divisible by pool_k");

    auto out = torch::empty({N, C, 1, 1, 1}, x.options());

    int total = N * C;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    fused_post_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, D, H, W,
        (int)pool_k,
        (float)scale,
        (float)clamp_min,
        (float)clamp_max
    );

    return out;
}
'''

_cpp_src = r'''
torch::Tensor fused_post_cuda(
    torch::Tensor x,
    int64_t pool_k,
    double scale,
    double clamp_min,
    double clamp_max
);
'''

_fused_mod = load_inline(
    name='fused_post_convtranspose3d_pool_avg_clamp_ext',
    cpp_sources=_cpp_src,
    cuda_sources=_cuda_src,
    functions=['fused_post_cuda'],
    extra_cuda_cflags=['-O3'],
    extra_cflags=['-O3'],
    verbose=False,
)

class ModelNew(nn.Module):
    """
    Optimized model:
    - Uses PyTorch ConvTranspose3d
    - Fuses scale * maxpool3d * global average pooling * clamp into one CUDA kernel
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.scale = float(scale)
        self.maxpool_kernel_size = int(maxpool_kernel_size)
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self._ext = _fused_mod

    def forward(self, x):
        x = self.conv_transpose(x)
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            return self._ext.fused_post_cuda(
                x, self.maxpool_kernel_size, self.scale, self.clamp_min, self.clamp_max
            )
        x = x * self.scale
        x = torch.nn.functional.max_pool3d(x, kernel_size=self.maxpool_kernel_size)
        x = torch.nn.functional.adaptive_avg_pool3d(x, (1, 1, 1))
        x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x