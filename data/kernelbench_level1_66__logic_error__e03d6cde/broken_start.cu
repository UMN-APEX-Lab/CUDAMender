import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void conv3d_forward_kernel(
    const float* input,
    const float* weight,
    float* output,
    int batch_size, int in_channels, int depth, int height, int width,
    int out_channels, int k_d, int k_h, int k_w,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    int dil_d, int dil_h, int dil_w,
    int out_depth, int out_height, int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_channels * out_depth * out_height * out_width;
    if (idx < total_elements) {
        int b = idx / (out_channels * out_depth * out_height * out_width);
        int rem = idx % (out_channels * out_depth * out_height * out_width);
        int oc = rem / (out_depth * out_height * out_width);
        rem %= (out_depth * out_height * out_width);
        int od = rem / (out_height * out_width);
        rem %= (out_height * out_width);
        int oh = rem / out_width;
        int ow = rem % out_width;

        float acc = 0.0f;
        int id_start = od * stride_d - pad_d;
        int ih_start = oh * stride_h - pad_h;
        int iw_start = ow * stride_w - pad_w;

        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kd = 0; kd < k_d; ++kd) {
                int id = id_start + kd * dil_d;
                if (id < 0 || id >= depth) continue;
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = ih_start + kh * dil_h;
                    if (ih < 0 || ih >= height) continue;
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = iw_start + kw * dil_w;
                        if (iw < 0 || iw >= width) continue;
                        
                        float inp = input[((b * in_channels + ic) * depth + id) * height * width + ih * width + iw];
                        float w = weight[((oc * in_channels + ic) * k_d + kd) * k_h * k_w + kh * k_w + kw];
                        acc += inp * w;
                    }
                }
            }
        }
        output[idx] = acc;
    }
}

torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, 
                                  int stride_d, int stride_h, int stride_w,
                                  int pad_d, int pad_h, int pad_w,
                                  int dil_d, int dil_h, int dil_w) {
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int depth = sizes[2];
    int height = sizes[3];
    int width = sizes[4];
    
    auto w_sizes = weight.sizes();
    int out_channels = w_sizes[0];
    int in_channels = w_sizes[1];
    int k_d = w_sizes[2];
    int k_h = w_sizes[3];
    int k_w = w_sizes[4];
    
    int out_depth = (depth + 2 * pad_d - dil_d * (k_d - 1) - 1) / stride_d + 1;
    int out_height = (height + 2 * pad_h - dil_h * (k_h - 1) - 1) / stride_h + 1;
    int out_width = (width + 2 * pad_w - dil_w * (k_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::empty({batch_size, out_channels, out_depth, out_height, out_width}, input.options());
    
    int total_elements = output.numel();
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    
    conv3d_forward_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(),
        batch_size, in_channels, depth, height, width,
        out_channels, k_d, k_h, k_w,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        dil_d, dil_h, dil_w,
        out_depth, out_height, out_width
    );
    
    return output;
}
"""

conv3d_cpp_source = """
torch::Tensor conv3d_forward_cuda(torch::Tensor input, torch::Tensor weight, 
                                  int stride_d, int stride_h, int stride_w,
                                  int pad_d, int pad_h, int pad_w,
                                  int dil_d, int dil_h, int dil_w);
"""

conv3d_op = load_inline(
    name='conv3d_op',
    cpp_sources=conv3d_cpp_source,
    cuda_sources=conv3d_cuda_source,
    functions=['conv3d_forward_cuda'],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple = (1, 1, 1), padding: tuple = (0, 0, 0), dilation: tuple = (1, 1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        weight = self.conv3d.weight.contiguous()
        
        out = conv3d_op.conv3d_forward_cuda(
            x, weight,
            self.stride[0], self.stride[1], self.stride[2],
            self.padding[0], self.padding[1], self.padding[2],
            self.dilation[0], self.dilation[1], self.dilation[2]
        )
        
        if self.bias:
            out = out + self.conv3d.bias.view(1, -1, 1, 1, 1)
            
        return out