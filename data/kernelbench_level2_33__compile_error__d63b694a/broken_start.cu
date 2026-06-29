import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel to fuse scaling and batch normalization
# BatchNorm1d(x) = (x - mean) / sqrt(var + eps) * gamma + beta
# Let x_scaled = x * scale
# The operation is: out = ((x * scale) - mean_scaled) / sqrt(var_scaled + eps) * gamma + beta
# To optimize, we can combine the scaling into the BN parameters or perform it element-wise.
# Given the architecture: x = (Linear(x) * scale), then BN(x).
# We implement a fused kernel that takes the GEMM output, applies the scale, and then applies BN.

fused_bn_scale_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_bn_scale_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ scale, 
    const float* __restrict__ running_mean, 
    const float* __restrict__ running_var, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ output, 
    int batch_size, 
    int out_features, 
    float eps) 
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && col < out_features) {
        int idx = row * out_features + col;
        float val = input[idx] * scale[col];
        float mean = running_mean[col];
        float var = running_var[col];
        float gamma = weight[col];
        float beta = bias[col];
        
        output[idx] = ((val - mean) / sqrtf(var + eps)) * gamma + beta;
    }
}

torch::Tensor fused_bn_scale_cuda(
    torch::Tensor input, 
    torch::Tensor scale, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float eps) 
{
    auto batch_size = input.size(0);
    auto out_features = input.size(1);
    auto output = torch::empty_like(input);

    dim3 block_size(16, 16);
    dim3 num_blocks((out_features + block_size.x - 1) / block_size.x, 
                    (batch_size + block_size.y - 1) / block_size.y);

    fused_bn_scale_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        scale.data_ptr<float>(), 
        running_mean.data_ptr<float>(), 
        running_var.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        output.data_ptr<float>(), 
        batch_size, 
        out_features, 
        eps);

    return output;
}
"""

fused_bn_scale_cpp_source = "torch::Tensor fused_bn_scale_cuda(torch::Tensor input, torch::Tensor scale, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, float eps);"

fused_op = load_inline(
    name="fused_bn_scale",
    cpp_sources=fused_bn_scale_cpp_source,
    cuda_sources=fused_bn_scale_source,
    functions=["fused_bn_scale_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        # GEMM is highly optimized in cuBLAS via torch.nn.Linear
        x = self.gemm(x)
        
        # Fuse scale and BatchNorm1d
        # We use the running statistics of the BN layer
        # Note: In training mode, BN updates stats. To maintain parity with nn.BatchNorm1d 
        # during training, we would need a more complex kernel. 
        # However, for the purpose of a speedup operator, we implement the functional 
        # application of the BN parameters.
        
        if self.training:
            # Fallback to standard PyTorch for training to handle running stats updates correctly
            x = x * self.scale
            x = self.bn(x)
        else:
            x = fused_op.fused_bn_scale_cuda(
                x, 
                self.scale, 
                self.bn.running_mean, 
                self.bn.running_var, 
                self.bn.weight, 
                self.bn.bias, 
                self.eps
            )
        return x