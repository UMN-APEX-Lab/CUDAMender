import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for RMSNorm
# Fused kernel: computes mean(x^2), sqrt(mean + eps), then x / rms all in one kernel
rmsnorm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused RMSNorm kernel: computes x / sqrt(mean(x^2) + eps)
// x: input tensor (N, C, H, W) where C is the feature dimension to normalize over
// out: output tensor
// gamma: optional scale parameter (not used in simple RMSNorm but kept for extensibility)
// N: batch size * H * W (total number of instances)
// C: number of features (channels)
// eps: small constant for numerical stability

__global__ void rmsnorm_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N,
    int C,
    float eps) {
    
    // Each block handles one instance (one position across all channels)
    // Block index corresponds to the instance index
    int instance_idx = blockIdx.x;
    
    if (instance_idx >= N) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    
    // Pointer to this instance's data
    const float* x_instance = x + instance_idx * C;
    float* out_instance = out + instance_idx * C;
    
    // Step 1: Compute sum of squares across all channels
    float thread_sum_sq = 0.0f;
    
    // Each thread handles multiple elements if C > blockDim.x
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = x_instance[c];
        thread_sum_sq += val * val;
    }
    
    // Store in shared memory
    shared_mem[threadIdx.x] = thread_sum_sq;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_mem[threadIdx.x] += shared_mem[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Compute RMS (root mean square)
    float mean_sq = shared_mem[0] / C;
    float rms = sqrtf(mean_sq + eps);
    
    // Step 2: Normalize each element
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        out_instance[c] = x_instance[c] / rms;
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps) {
    // x shape: (batch_size, num_features, dim1, dim2)
    // We normalize over dim=1 (num_features)
    
    auto batch_size = x.size(0);
    auto num_features = x.size(1);
    auto dim1 = x.size(2);
    auto dim2 = x.size(3);
    
    // Total number of instances: batch_size * dim1 * dim2
    auto N = batch_size * dim1 * dim2;
    auto C = num_features;
    
    // Create output tensor
    auto out = torch::empty_like(x);
    
    // Configure kernel
    const int threads = 256;  // Threads per block for reduction
    const int blocks = N;     // One block per instance
    
    // Shared memory size for reduction
    size_t shared_mem_size = threads * sizeof(float);
    
    rmsnorm_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        C,
        eps
    );
    
    return out;
}
"""

rmsnorm_cpp_source = "torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps);"

# Compile the inline CUDA code
rmsnorm_cuda = load_inline(
    name="rmsnorm_cuda",
    cpp_sources=rmsnorm_cpp_source,
    cuda_sources=rmsnorm_source,
    functions=["rmsnorm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized RMS Normalization using custom fused CUDA kernel.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use fused CUDA kernel that combines:
        # - mean(x^2) computation
        # - sqrt(mean + eps)
        # - division
        # All in a single kernel launch, avoiding multiple memory passes
        return rmsnorm_cuda.rmsnorm_cuda(x, self.eps)