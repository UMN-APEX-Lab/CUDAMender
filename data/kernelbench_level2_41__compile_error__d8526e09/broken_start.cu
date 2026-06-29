import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# CUDA kernel for fusing BatchNorm (inference mode), GELU, and ReLU
# BatchNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
# GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# ReLU: max(0, x)
# Note: GELU(x) is always >= 0 for x >> 0, but for x < 0 it can be negative.
# However, ReLU(GELU(x)) effectively clips the negative part of GELU.

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ inline float gelu_relu(float x) {
    // GELU approximation
    float x3 = x * x * x;
    float gelu = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
    // ReLU
    return gelu > 0.0f ? gelu : 0.0f;
}

__global__ void fused_bn_gelu_relu_kernel(
    const float* __restrict__ x, 
    const float* __restrict__ running_mean, 
    const float* __restrict__ running_var, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* __restrict__ out, 
    int batch_size, 
    int out_features, 
    float eps) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * out_features) {
        int b = idx / out_features;
        int f = idx % out_features;
        
        float val = x[idx];
        float mean = running_mean[f];
        float var = running_var[f];
        float w = weight[f];
        float b_val = bias[f];
        
        // BatchNorm
        float norm = (val - mean) * rsqrtf(var + eps);
        float bn_out = norm * w + b_val;
        
        // GELU + ReLU
        out[idx] = gelu_relu(bn_out);
    }
}

torch::Tensor fused_bn_gelu_relu_cuda(
    torch::Tensor x, 
    torch::Tensor running_mean, 
    torch::Tensor running_var, 
    torch::Tensor weight, 
    torch::Tensor bias, 
    float eps) 
{
    auto batch_size = x.size(0);
    auto out_features = x.size(1);
    auto out = torch::empty_like(x);

    const int block_size = 256;
    const int total_elements = batch_size * out_features;
    const int num_blocks = (total_elements + block_size - 1) / block_size;

    fused_bn_gelu_relu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), 
        running_mean.data_ptr<float>(), 
        running_var.data_ptr<float>(), 
        weight.data_ptr<float>(), 
        bias.data_ptr<float>(), 
        out.data_ptr<float>(), 
        batch_size, 
        out_features, 
        eps
    );

    return out;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_bn_gelu_relu_cuda(torch::Tensor x, torch::Tensor running_mean, torch::Tensor running_var, torch::Tensor weight, torch::Tensor bias, float eps);"
)

fused_op = load_inline(
    name="fused_bn_gelu_relu",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_bn_gelu_relu_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        # We use the weights and buffers from batch_norm in the fused kernel

    def forward(self, x):
        # GEMM is highly optimized in cuBLAS, keeping it as is.
        x = self.gemm(x)
        
        # Fuse BatchNorm (eval mode), GELU, and ReLU
        # Note: To use the fused kernel, we assume the model is in eval mode for BN.
        # If training is needed, this kernel would need gradients and mean/var calculation.
        if not self.training:
            return fused_op.fused_bn_gelu_relu_cuda(
                x, 
                self.batch_norm.running_mean, 
                self.batch_norm.running_var, 
                self.batch_norm.weight, 
                self.batch_norm.bias, 
                self.batch_norm.eps
            )
        else:
            # Fallback for training mode
            x = self.batch_norm(x)
            x = torch.nn.functional.gelu(x)
            x = torch.relu(x)
            return x