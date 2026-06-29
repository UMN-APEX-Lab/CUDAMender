import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Linear + Mish + Mish
fused_linear_double_mish_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_linear_double_mish_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < batch_size && col < out_features) {
        // Compute linear: y = x @ W^T + b
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        if (bias != nullptr) {
            sum += bias[col];
        }
        
        // First Mish activation: x * tanh(softplus(x))
        // softplus(x) = log(1 + exp(x))
        float sp = logf(1.0f + expf(sum));
        float tanh_sp = tanhf(sp);
        float mish1 = sum * tanh_sp;
        
        // Second Mish activation
        float sp2 = logf(1.0f + expf(mish1));
        float tanh_sp2 = tanhf(sp2);
        float mish2 = mish1 * tanh_sp2;
        
        output[row * out_features + col] = mish2;
    }
}

torch::Tensor fused_linear_double_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias
) {
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    const int BLOCK_SIZE_X = 16;
    const int BLOCK_SIZE_Y = 16;
    
    dim3 block_size(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_size(
        (out_features + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (batch_size + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y
    );
    
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    fused_linear_double_mish_kernel<<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features
    );
    
    return output;
}
"""

fused_linear_double_mish_cpp_source = """
torch::Tensor fused_linear_double_mish_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias
);
"""

# Compile the inline CUDA code
fused_linear_double_mish = load_inline(
    name="fused_linear_double_mish",
    cpp_sources=fused_linear_double_mish_cpp_source,
    cuda_sources=fused_linear_double_mish_source,
    functions=["fused_linear_double_mish_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that fuses Linear + Mish + Mish into a single CUDA kernel.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        self.fused_op = fused_linear_double_mish

    def forward(self, x):
        return self.fused_op.fused_linear_double_mish_cuda(x, self.weight, self.bias)


import math

batch_size = 1024
in_features = 8192
out_features = 8192

def get_inputs():
    return [torch.rand(batch_size, in_features).cuda()]

def get_init_inputs():
    return [in_features, out_features]