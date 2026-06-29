import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_depthwise_conv_ext = None


def _load_depthwise_conv_ext():
    global _depthwise_conv_ext
    if _depthwise_conv_ext is not None:
        return _depthwise_conv_ext
    if not torch.cuda.is_available():
        return None

    cpp_source = r"""
    torch::Tensor depthwise_conv2d_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int64_t stride_h,
        int64_t stride_w,
        int64_t pad_h,
        int64_t pad_w,
        int64_t dilation_h,
        int64_t dilation_w
    );
    """

    cuda_source = r"""
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <vector>

    #define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
    #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
    #define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, #x " must be float32")

    __global__ void depthwise_conv2d_kernel(
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias,
        float* __restrict__ output,
        int N, int C, int H, int W,
        int K_H, int K_W,
        int out_H, int out_W,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int dilation_h, int dilation_w,
        bool has_bias
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = N * C * out_H * out_W;
        if (idx >= total) return;

        int ow = idx % out_W;
        int oh = (idx / out_W) % out_H;
        int c = (idx / (out_W * out_H)) % C;
        int n = idx / (out_W * out_H * C);

        float sum = has_bias ? bias[c] : 0.0f;

        const int input_nc_offset = ((n * C + c) * H) * W;
        const int weight_c_offset = c * K_H * K_W;

        for (int kh = 0; kh < K_H; ++kh) {
            int ih = oh * stride_h - pad_h + kh * dilation_h;
            if ((unsigned)ih >= (unsigned)H) continue;
            for (int kw = 0; kw < K_W; ++kw) {
                int iw = ow * stride_w - pad_w + kw * dilation_w;
                if ((unsigned)iw >= (unsigned)W) continue;
                float in_val = input[input_nc_offset + ih * W + iw];
                float w_val = weight[weight_c_offset + kh * K_W + kw];
                sum += in_val * w_val;
            }
        }

        output[idx] = sum;
    }

    torch::Tensor depthwise_conv2d_cuda(
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        int64_t stride_h,
        int64_t stride_w,
        int64_t pad_h,
        int64_t pad_w,
        int64_t dilation_h,
        int64_t dilation_w
    ) {
        CHECK_CUDA(input);
        CHECK_CUDA(weight);
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(weight);
        CHECK_FLOAT(input);
        CHECK_FLOAT(weight);
        TORCH_CHECK(input.dim() == 4, "input must be 4D NCHW");
        TORCH_CHECK(weight.dim() == 4, "weight must be 4D [C,1,KH,KW]");
        TORCH_CHECK(weight.size(1) == 1, "depthwise weight second dim must be 1");

        const auto N = (int)input.size(0);
        const auto C = (int)input.size(1);
        const auto H = (int)input.size(2);
        const auto W = (int)input.size(3);

        TORCH_CHECK(weight.size(0) == C, "weight.size(0) must equal input channels");

        const auto K_H = (int)weight.size(2);
        const auto K_W = (int)weight.size(3);

        const int out_H = (H + 2 * (int)pad_h - (int)dilation_h * (K_H - 1) - 1) / (int)stride_h + 1;
        const int out_W = (W + 2 * (int)pad_w - (int)dilation_w * (K_W - 1) - 1) / (int)stride_w + 1;

        TORCH_CHECK(out_H >= 0 && out_W >= 0, "invalid output size");

        auto output = torch::empty({N, C, out_H, out_W}, input.options());

        bool has_bias = bias.defined() && bias.numel() > 0;
        const float* bias_ptr = nullptr;

        if (has_bias) {
            CHECK_CUDA(bias);
            CHECK_CONTIGUOUS(bias);
            CHECK_FLOAT(bias);
            TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "bias must have shape [C]");
            bias_ptr = bias.data_ptr<float>();
        }

        const int total = N * C * out_H * out_W;
        const int threads = 256;
        const int blocks = (total + threads - 1) / threads;

        depthwise_conv2d_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias_ptr,
            output.data_ptr<float>(),
            N, C, H, W,
            K_H, K_W,
            out_H, out_W,
            (int)stride_h, (int)stride_w,
            (int)pad_h, (int)pad_w,
            (int)dilation_h, (int)dilation_w,
            has_bias
        );

        return output;
    }
    """

    try:
        _depthwise_conv_ext = load_inline(
            name="depthwise_conv2d_ext_v1",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["depthwise_conv2d_cuda"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    except Exception:
        _depthwise_conv_ext = None
    return _depthwise_conv_ext


class _DepthwiseConvCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, fallback_module):
        ext = _load_depthwise_conv_ext()
        if (
            ext is None
            or (not x.is_cuda)
            or x.dtype != torch.float32
            or weight.dtype != torch.float32
            or (bias is not None and bias.dtype != torch.float32)
        ):
            return fallback_module(x)

        x_c = x.contiguous()
        w_c = weight.contiguous()
        b_c = bias.contiguous() if bias is not None else torch.empty(0, device=x.device, dtype=x.dtype)

        return ext.depthwise_conv2d_cuda(
            x_c,
            w_c,
            b_c,
            int(stride[0]),
            int(stride[1]),
            int(padding[0]),
            int(padding[1]),
            int(dilation[0]),
            int(dilation[1]),
        )


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _DepthwiseConvCudaFunction.apply(
            x,
            self.depthwise.weight,
            self.depthwise.bias,
            self.depthwise.stride,
            self.depthwise.padding,
            self.depthwise.dilation,
            self.depthwise,
        )
        x = self.pointwise(x)
        return x