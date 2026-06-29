import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_fused_ext = None

if torch.cuda.is_available():
    _cpp_src = r"""
    #include <torch/extension.h>
    torch::Tensor fused_post_conv_cuda(torch::Tensor x, torch::Tensor bias);
    """

    _cuda_src = r"""
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <vector>
    #include <cmath>

    __device__ __forceinline__ float gelu_approx(float x) {
        const float kAlpha = 0.7978845608028654f; // sqrt(2/pi)
        const float kBeta = 0.044715f;
        float x3 = x * x * x;
        return 0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x3)));
    }

    __global__ void fused_post_conv_kernel(
        const float* __restrict__ x,
        const float* __restrict__ bias,
        float* __restrict__ out,
        int64_t total,
        int64_t C,
        int64_t inner_size
    ) {
        int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total) return;

        int64_t c = (idx / inner_size) % C;
        float v = x[idx];

        v = v > 0.0f ? v : 0.0f;                 // ReLU
        v = v > 0.0f ? v : 0.01f * v;            // LeakyReLU
        v = gelu_approx(v);                      // GELU
        v = 1.0f / (1.0f + expf(-v));            // Sigmoid
        v += bias[c];                            // Bias add

        out[idx] = v;
    }

    torch::Tensor fused_post_conv_cuda(torch::Tensor x, torch::Tensor bias) {
        TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
        TORCH_CHECK(bias.scalar_type() == at::kFloat, "bias must be float32");
        TORCH_CHECK(x.dim() == 5, "x must be 5D NCDHW");
        TORCH_CHECK(bias.numel() == x.size(1), "bias must have one value per output channel");

        auto x_contig = x.contiguous();
        auto bias_contig = bias.contiguous().view({-1});
        auto out = torch::empty_like(x_contig);

        const int64_t total = x_contig.numel();
        const int64_t C = x_contig.size(1);
        const int64_t inner_size = x_contig.size(2) * x_contig.size(3) * x_contig.size(4);

        const int threads = 256;
        const int blocks = static_cast<int>((total + threads - 1) / threads);

        fused_post_conv_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            x_contig.data_ptr<float>(),
            bias_contig.data_ptr<float>(),
            out.data_ptr<float>(),
            total,
            C,
            inner_size
        );

        return out;
    }
    """

    try:
        _fused_ext = load_inline(
            name="fused_post_conv_ext",
            cpp_sources=_cpp_src,
            cuda_sources=_cuda_src,
            functions=["fused_post_conv_cuda"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        _fused_ext = None


def _fallback_fused_post_conv(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = torch.relu(x)
    x = F.leaky_relu(x, negative_slope=0.01)
    x = F.gelu(x)
    x = torch.sigmoid(x)
    x = x + bias
    return x


class ModelNew(nn.Module):
    """
    Optimized model: keeps Conv3d in PyTorch/cuDNN and fuses
    ReLU -> LeakyReLU -> GELU -> Sigmoid -> BiasAdd into one CUDA kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape, dtype=torch.float32))

    def forward(self, x):
        x = self.conv(x)
        if (
            _fused_ext is not None
            and x.is_cuda
            and self.bias.is_cuda
            and x.dtype == torch.float32
            and self.bias.dtype == torch.float32
        ):
            return _fused_ext.fused_post_conv_cuda(x, self.bias)
        return _fallback_fused_post_conv(x, self.bias)


batch_size = 64
in_channels = 8
out_channels = 32
depth, height, width = 32, 64, 64
kernel_size = 3
bias_shape = (out_channels, 1, 1, 1)


def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width, dtype=torch.float32)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]