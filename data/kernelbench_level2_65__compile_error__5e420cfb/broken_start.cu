import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# We optimize the post-convolution sequence: AvgPool2d -> Sigmoid -> Sum.
# Since AvgPool2d is a linear operation and Sum is linear, we can combine them.
# Sum(AvgPool(x)) = Sum(x) / (pool_kernel_size^2).
# However, the Sigmoid is non-linear and sits between them.
# The operation is: Sum_{c, h, w} sigmoid( (1/k^2) * Sum_{i,j} x[c, h+i, w+j] )
# This is a fused kernel that computes the average pool, applies sigmoid, and accumulates the sum per batch item.

fused_pool_sig_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_pool_sig_sum_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int batch_size, int channels, int in_h, int in_w, 
    int pool_h, int pool_w, int out_h, int out_w) 
{
    int b = blockIdx.z;
    int c = blockIdx.y;
    int oh = blockIdx.x / (out_w); // simplified indexing
    int ow = blockIdx.x % (out_w);
    
    // We use a different mapping for better parallelism
    // Let's use blockIdx.x for the output spatial location (oh, ow)
}
"""

# Redefining the kernel for better performance and correctness
fused_pool_sig_sum_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_kernel(
    const float* __restrict__ input, 
    float* __restrict__ output, 
    int batch_size, int channels, int in_h, int in_w, 
    int pool_h, int pool_w, int out_h, int out_w) 
{
    int b = blockIdx.z;
    int c = blockIdx.y;
    int oh = blockIdx.x / out_w;
    int ow = blockIdx.x % out_w;

    if (oh >= out_h || ow >= out_w) return;

    float sum = 0.0f;
    for (int i = 0; i < pool_h; ++i) {
        for (int j = 0; j < pool_w; ++j) {
            int ih = oh * pool_h + i;
            int iw = ow * pool_w + j;
            sum += input[((b * channels + c) * in_h + ih) * in_w + iw];
        }
    }
    
    float avg = sum / (float)(pool_h * pool_w);
    float sig = 1.0f / (1.0f + expf(-avg));
    
    // Atomic add to the batch result
    atomicAdd(&output[b], sig);
}

torch::Tensor fused_pool_sig_sum_cuda(torch::Tensor input, int pool_h, int pool_w) {
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto in_h = input.size(2);
    auto in_w = input.size(3);
    
    auto out_h = in_h / pool_h;
    auto out_w = in_w / pool_w;
    
    auto output = torch::zeros({batch_size}, input.options());
    
    dim3 block(1, 1, 1); // Simple mapping for clarity, can be optimized
    dim3 grid(out_h * out_w, channels, batch_size);
    
    fused_kernel<<<grid, block>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, channels, in_h, in_w, 
        pool_h, pool_w, out_h, out_w
    );
    
    return output;
}
"""

fused_pool_sig_sum_cpp_source = "torch::Tensor fused_pool_sig_sum_cuda(torch::Tensor input, int pool_h, int pool_w);"

fused_op = load_inline(
    name="fused_pool_sig_sum",
    cpp_sources=fused_pool_sig_sum_cpp_source,
    cuda_sources=fused_pool_sig_sum_source,
    functions=["fused_pool_sig_sum_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        # Fuse AvgPool2d -> Sigmoid -> Sum
        return fused_op.fused_pool_sig_sum_cuda(x, self.pool_kernel_size, self.pool_kernel_size)