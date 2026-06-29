import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused Linear + ReLU + Division CUDA kernel
fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void fused_linear_relu_div_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int M,
    int K,
    float divisor
) {
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= N || out_idx >= K) return;
    
    float sum = 0.0f;
    if (bias != nullptr) {
        sum = bias[out_idx];
    }
    
    #pragma unroll 4
    for (int i = 0; i < M; i++) {
        sum += input[batch_idx * M + i] * weight[out_idx * M + i];
    }
    
    sum = fmaxf(0.0f, sum);
    sum /= divisor;
    
    output[batch_idx * K + out_idx] = sum;
}

torch::Tensor fused_linear_relu_div_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float divisor
) {
    int N = input.size(0);
    int M = input.size(1);
    int K = weight.size(0);
    
    auto output = torch::empty({N, K}, input.options());
    
    dim3 block_size(16, 16);
    dim3 grid_size((K + block_size.x - 1) / block_size.x,
                   (N + block_size.y - 1) / block_size.y);
    
    fused_linear_relu_div_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, M, K, divisor
    );
    
    return output;
}
"""

fused_op_cpp = "torch::Tensor fused_linear_relu_div_cuda(torch::Tensor, torch::Tensor, torch::Tensor, float);"

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp,
    cuda_sources=fused_op_source,
    functions=["fused_linear_relu_div_cuda"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, divisor):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.divisor = divisor

    def forward(self, x):
        weight = self.linear.weight
        bias = self.linear.bias
        return fused_op.fused_linear_relu_div_cuda(x, weight, bias, self.divisor)