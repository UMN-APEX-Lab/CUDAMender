import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_op_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void fused_groupnorm_min_bias_kernel(
    const float* input, float* output,
    const float* gamma, const float* beta,
    const float* bias,
    int batch_size, int channels, int num_groups,
    float eps) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= batch_size) return;
    
    int batch_id = tid;
    int features_per_group = channels / num_groups;
    
    // Process each group for this batch
    for (int g = 0; g < num_groups; g++) {
        int group_start = g * features_per_group;
        
        // Calculate mean for this group
        float sum = 0.0f;
        for (int f = 0; f < features_per_group; f++) {
            int idx = batch_id * channels + group_start + f;
            sum += input[idx];
        }
        float mean = sum / features_per_group;
        
        // Calculate variance for this group
        float sum_sq = 0.0f;
        for (int f = 0; f < features_per_group; f++) {
            int idx = batch_id * channels + group_start + f;
            float diff = input[idx] - mean;
            sum_sq += diff * diff;
        }
        float var = sum_sq / features_per_group;
        float rstd = 1.0f / sqrtf(var + eps);
        
        // Find minimum normalized value in this group
        for (int f = 0; f < features_per_group; f++) {
            int idx = batch_id * channels + group_start + f;
            float normalized = (input[idx] - mean) * rstd;
            
            // Apply gamma and beta
            float gn_out = normalized * gamma[group_start + f] + beta[group_start + f];
            
            if (gn_out < output[batch_id]) {
                output[batch_id] = gn_out;
            }
        }
    }
    
    // Add bias
    output[batch_id] += bias[batch_id % 1];
}

torch::Tensor fused_groupnorm_min_bias_cuda(
    const torch::Tensor& input,
    const torch::Tensor& gamma,
    const torch::Tensor& beta,
    const torch::Tensor& bias,
    int num_groups,
    float eps) {
    
    auto batch_size = input.size(0);
    auto channels = input.size(1);
    
    auto output = torch::empty({batch_size, 1}, input.options());
    
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;
    
    fused_groupnorm_min_bias_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size, channels, num_groups, eps
    );
    
    return output;
}
"""

fused_op_cpp_source = (
    "torch::Tensor fused_groupnorm_min_bias_cuda("
    "const torch::Tensor& input,"
    "const torch::Tensor& gamma,"
    "const torch::Tensor& beta,"
    "const torch::Tensor& bias,"
    "int num_groups,"
    "float eps);"
)

fused_op = load_inline(
    name="fused_op",
    cpp_sources=fused_op_cpp_source,
    cuda_sources=fused_op_source,
    functions=["fused_groupnorm_min_bias_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.num_groups = num_groups
        self.eps = 1e-5
        
        # GroupNorm parameters (gamma and beta)
        self.gamma = nn.Parameter(torch.ones(out_features))
        self.beta = nn.Parameter(torch.zeros(out_features))
        
        # Bias parameter - reshape to match output after min operation
        self.bias = nn.Parameter(torch.randn(1))
        
        # Register fused operation
        self.fused_op = fused_op

    def forward(self, x):
        # GEMM (keep as PyTorch for cuBLAS optimization)
        x = self.gemm(x)
        
        # Fused GroupNorm + Min + Bias
        x = self.fused_op.fused_groupnorm_min_bias_cuda(
            x, self.gamma, self.beta, self.bias, self.num_groups, self.eps
        )
        
        return x

def get_inputs():
    return [torch.rand(1024, 8192)]

def get_init_inputs():
    return [8192, 8192, 512, (1, 8192, 1, 1)]