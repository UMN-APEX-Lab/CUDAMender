import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# The core bottleneck after Conv3d is GroupNorm followed by a global mean.
# GroupNorm calculates mean and variance per group per sample.
# We can fuse the GroupNorm normalization and the final global mean into a single kernel
# to avoid writing the large normalized tensor back to VRAM and reading it again for the mean.
# However, since the final operation is a mean over all dimensions (C, D, H, W),
# we can simplify the math:
# Mean(GroupNorm(x)) = Mean( (x - mean_g) / sqrt(var_g + eps) * gamma_g + beta_g )
# This is still complex because GroupNorm is per-group. 
# To maximize speedup and maintain correctness, we implement a fused GroupNorm + GlobalMean kernel.

fused_gn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_gn_mean_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int G, float eps) {
    
    int n = blockIdx.x;
    if (n >= N) return;

    int channels_per_group = C / G;
    int elements_per_group = channels_per_group * D * H * W;
    
    double total_sum = 0.0;

    for (int g = 0; g < G; ++g) {
        double group_sum = 0.0;
        double group_sq_sum = 0.0;
        
        int group_offset = g * channels_per_group * D * H * W;
        
        for (int i = 0; i < elements_per_group; ++i) {
            float val = input[n * C * D * H * W + group_offset + i];
            group_sum += val;
            group_sq_sum += (double)val * val;
        }
        
        float mean = group_sum / elements_per_group;
        float var = (group_sq_sum / elements_per_group) - (mean * mean);
        float inv_std = 1.0f / sqrtf(var + eps);
        
        // Now calculate the sum of normalized values for this group
        // GroupNorm: y = (x - mean) * inv_std * gamma + beta
        // Sum(y) = Sum((x - mean) * inv_std * gamma) + Sum(beta)
        // Since gamma and beta are per-channel:
        for (int c = 0; c < channels_per_group; ++c) {
            float g_gamma = gamma[g * channels_per_group + c];
            float g_beta = beta[g * channels_per_group + c];
            
            double channel_sum = 0.0;
            int channel_offset = group_offset + c * D * H * W;
            for (int i = 0; i < D * H * W; ++i) {
                channel_sum += (input[n * C * D * H * W + channel_offset + i] - mean) * inv_std;
            }
            total_sum += (channel_sum * g_gamma) + (double)g_beta * D * H * W;
        }
    }
    
    output[n] = (float)(total_sum / (C * D * H * W));
}

torch::Tensor fused_gn_mean_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    const int N = input.size(0);
    const int C = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    const int G = gamma.size(0) / (C / (gamma.size(0) / (C / (C / (C/C)))); // This is just a placeholder, we need G
    // Wait, G is passed as an argument to the model. We'll get it from the gamma/beta shape or a passed int.
    // Correct logic: G is known from the GroupNorm config.
    return torch::empty({N}, input.options());
}
"""

# Redefining the source to be more robust and handle G correctly
fused_gn_mean_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void fused_gn_mean_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int G, float eps) {
    
    int n = blockIdx.x;
    if (n >= N) return;

    int channels_per_group = C / G;
    int elements_per_group = channels_per_group * D * H * W;
    double total_sum = 0.0;

    for (int g = 0; g < G; ++g) {
        double group_sum = 0.0;
        double group_sq_sum = 0.0;
        int group_offset = g * channels_per_group * D * H * W;
        
        for (int i = 0; i < elements_per_group; ++i) {
            float val = input[n * C * D * H * W + group_offset + i];
            group_sum += val;
            group_sq_sum += (double)val * val;
        }
        
        float mean = group_sum / elements_per_group;
        float var = (group_sq_sum / elements_per_group) - (mean * mean);
        float inv_std = 1.0f / sqrtf(var + eps);
        
        for (int c = 0; c < channels_per_group; ++c) {
            float g_gamma = gamma[g * channels_per_group + c];
            float g_beta = beta[g * channels_per_group + c];
            double channel_sum = 0.0;
            int channel_offset = group_offset + c * D * H * W;
            for (int i = 0; i < D * H * W; ++i) {
                channel_sum += (input[n * C * D * H * W + channel_offset + i] - mean) * inv_std;
            }
            total_sum += (channel_sum * g_gamma) + (double)g_beta * D * H * W;
        }
    }
    output[n] = (float)(total_sum / (C * D * H * W));
}

torch::Tensor fused_gn_mean_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int G, float eps) {
    auto N = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    auto output = torch::empty({N}, input.options());

    fused_gn_mean_kernel<<<N, 1>>>(
        input.data_ptr<float>(), 
        gamma.data_ptr<float>(), 
        beta.data_ptr<float>(), 
        output.data_ptr<float>(), 
        N, C, D, H, W, G, eps
    );

    return output;
}
"""

fused_gn_mean_cpp_source = "torch::Tensor fused_gn_mean_cuda(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, int G, float eps);"

fused_op = load_inline(
    name="fused_gn_mean",
    cpp_sources=fused_gn_mean_cpp_source,
    cuda_sources=fused_gn_mean_source,
    functions=["fused_gn_mean_cuda"],
    verbose=False,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups

    def forward(self, x):
        x = self.conv(x)
        # Fuse GroupNorm and Mean
        # GroupNorm weights are gamma (weight) and beta (bias)
        return fused_op.fused_gn_mean_cuda(
            x, 
            self.group_norm.weight, 
            self.group_norm.bias, 
            self.num_groups, 
            self.group_norm.eps
        )