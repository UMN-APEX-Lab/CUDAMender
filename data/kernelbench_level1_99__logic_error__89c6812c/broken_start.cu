import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for Triplet Margin Loss
# This fuses: distance computation (L2), margin-based loss, and mean reduction
triplet_margin_loss_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void triplet_margin_loss_kernel(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* loss_per_sample,
    int batch_size,
    int dim,
    float margin) {
    
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample_idx >= batch_size) return;
    
    float pos_dist_sq = 0.0f;
    float neg_dist_sq = 0.0f;
    
    // Compute L2 distances squared for positive and negative pairs
    #pragma unroll 8
    for (int i = 0; i < dim; i++) {
        float a = anchor[sample_idx * dim + i];
        float p = positive[sample_idx * dim + i];
        float n = negative[sample_idx * dim + i];
        
        float diff_pos = a - p;
        float diff_neg = a - n;
        
        pos_dist_sq += diff_pos * diff_pos;
        neg_dist_sq += diff_neg * diff_neg;
    }
    
    // Compute triplet loss: max(d(a,p) - d(a,n) + margin, 0)
    // Using sqrt for proper distance, but we can optimize by comparing squared distances
    // However, TripletMarginLoss uses p-norm distances, so we need sqrt
    float pos_dist = sqrtf(pos_dist_sq);
    float neg_dist = sqrtf(neg_dist_sq);
    
    float loss = pos_dist - neg_dist + margin;
    loss_per_sample[sample_idx] = fmaxf(loss, 0.0f);
}

__global__ void reduce_mean_kernel(const float* input, float* output, int n) {
    // Simple block-wise reduction for mean
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    float val = 0.0f;
    if (idx < n) {
        val = input[idx];
    }
    sdata[tid] = val;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {
    
    int batch_size = anchor.size(0);
    int dim = anchor.size(1);
    
    // Ensure contiguous memory layout
    anchor = anchor.contiguous();
    positive = positive.contiguous();
    negative = negative.contiguous();
    
    // Allocate output for per-sample losses
    auto loss_per_sample = torch::empty({batch_size}, anchor.options());
    
    // Compute per-sample losses
    const int threads = 256;
    const int blocks = (batch_size + threads - 1) / threads;
    
    triplet_margin_loss_kernel<<<blocks, threads>>>(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        loss_per_sample.data_ptr<float>(),
        batch_size,
        dim,
        margin);
    
    // Compute mean using efficient reduction
    auto total_loss = torch::zeros({1}, anchor.options());
    
    reduce_mean_kernel<<<1, threads, threads * sizeof(float)>>>(
        loss_per_sample.data_ptr<float>(),
        total_loss.data_ptr<float>(),
        batch_size);
    
    return total_loss / batch_size;
}
"""

triplet_margin_loss_cpp_source = """
torch::Tensor triplet_margin_loss_cuda(torch::Tensor anchor, torch::Tensor positive, torch::Tensor negative, float margin);
"""

# Compile the inline CUDA code
triplet_margin_loss = load_inline(
    name="triplet_margin_loss",
    cpp_sources=triplet_margin_loss_cpp_source,
    cuda_sources=triplet_margin_loss_source,
    functions=["triplet_margin_loss_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that computes Triplet Margin Loss using custom CUDA kernel.
    Fuses distance computation, margin application, and reduction into a single kernel.
    """
    def __init__(self, margin=1.0):
        super(ModelNew, self).__init__()
        self.margin = margin
        self.loss_fn = triplet_margin_loss

    def forward(self, anchor, positive, negative):
        return self.loss_fn.triplet_margin_loss_cuda(anchor, positive, negative, self.margin).squeeze()