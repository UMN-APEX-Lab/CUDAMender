import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv_transpose3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int B, int IC, int OC, int ID, int IH, int IW,
    int KD, int KH, int KW,
    int OD, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * OC * OD * OH * OW;
    if (idx >= total) return;

    int rem = idx;
    int ow = rem % OW; rem /= OW;
    int oh = rem % OH; rem /= OH;
    int od = rem % OD; rem /= OD;
    int oc = rem % OC; rem /= OC;
    int b = rem;

    float sum = 0.0f;
    for (int ic = 0; ic < IC; ++ic) {
        for (int kd = 0; kd < KD; ++kd) {
            int id = od - kd;
            if (id < 0 || id >= ID) continue;
            for (int kh = 0; kh < KH; ++kh) {
                int ih = oh - kh;
                if (ih < 0 || ih >= IH) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    int iw = ow - kw;
                    if (iw < 0 || iw >= IW) continue;

                    int in_idx = (((b * IC + ic) * ID + id) * IH + ih) * IW + iw;
                    int w_idx = (((ic * OC + oc) * KD + kd) * KH + kh) * KW + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    int out_idx = (((b * OC + oc) * OD + od) * OH + oh) * OW + ow;
    output[out_idx] = sum;
}

torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight) {
    input = input.contiguous();
    weight = weight.contiguous();

    int B = input.size(0);
    int IC = input.size(1);
    int ID = input.size(2);
    int IH = input.size(3);
    int IW = input.size(4);

    int OC = weight.size(1);
    int KD = weight.size(2);
    int KH = weight.size(3);
    int KW = weight.size(4);

    int OD = ID + KD - 1;
    int OH = IH + KH - 1;
    int OW = IW + KW - 1;

    auto output = torch::zeros({B, OC, OD, OH, OW}, input.options());

    const int total = B * OC * OD * OH * OW;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    conv_transpose3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B, IC, OC, ID, IH, IW, KD, KH, KW, OD, OH, OW
    );

    return output;
}
"""

cpp_source = "torch::Tensor conv_transpose3d_cuda(torch::Tensor input, torch::Tensor weight);"

conv_transpose3d_op = load_inline(
    name="conv_transpose3d_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conv_transpose3d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), output_padding: tuple = (0, 0, 0), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv_transpose3d = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_transpose3d_op.conv_transpose3d_cuda(x, self.conv_transpose3d.weight)