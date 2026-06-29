import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel to fuse Dropout and Softmax
# Dropout is applied as x = x * (1/(1-p)) * mask
# Softmax is applied over the last dimension
dropout_softmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

__global__ void dropout_softmax_kernel(float* data, float p, float scale, int batch_size, int out_features, unsigned long long seed) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < batch_size && col < out_features) {
        int idx = row * out_features + col;
        
        // Dropout part
        curandStatePhilox4_32_10_t state;
        curand_init(seed, idx, 0, &state);
        float rand_val = curand_uniform(&state);
        
        float val = data[idx];
        if (rand_val < p) {
            val = 0.0f;
        } else {
            val *= scale;
        }
        data[idx] = val;
    }
}

__global__ void softmax_kernel(float* data, int batch_size, int out_features) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    float max_val = -1e38f;
    for (int col = 0; col < out_features; ++col) {
        max_val = fmaxf(max_val, data[row * out_features + col]);
    }

    float sum = 0.0f;
    for (int col = 0; col < out_features; ++col) {
        float val = expf(data[row * out_features + col] - max_val);
        data[row * out_features + col] = val;
        sum += val;
    }

    float inv_sum = 1.0f / sum;
    for (int col = 0; col < out_features; ++col) {
        data[row * out_features + col] *= inv_sum;
    }
}

torch::Tensor dropout_softmax_cuda(torch::Tensor x, float p, bool training) {
    if (!training) {
        // Just softmax if not training
        auto out = x.clone();
        int batch_size = out.size(0);
        int out_features = out.size(1);
        softmax_kernel<<<batch_size, 1>>>(out.data_ptr<float>(), batch_size, out_features);
        return out;
    }

    auto out = x.clone();
    int batch_size = out.size(0);
    int out_features = out.size(1);
    float scale = 1.0f / (1.0f - p);
    unsigned long long seed = 1234ULL;

    // Since out_features is large (16384), we need to handle block size limits
    // We use a simpler approach for dropout to ensure it fits in block size or loop
    // For this specific task, we use a kernel that handles columns via loops or multiple blocks
    // Redefining dropout kernel for larger out_features
    return out; 
}
"""

# Since the provided architecture has very large out_features (16384), 
# a simple 1D block for softmax is inefficient. 
# We'll use a more robust fused kernel for Dropout + Softmax.

fused_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void fused_dropout_softmax_kernel(float* data, float p, float scale, int batch_size, int out_features, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * out_features;
    
    // 1. Dropout
    if (tid < total_elements) {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, tid, 0, &state);
        if (curand_uniform(&state) < p) {
            data[tid] = 0.0f;
        } else {
            data[tid] *= scale;
        }
    }
    __syncthreads(); // This doesn't work across blocks, but dropout is element-wise.
}

// To properly fuse, we perform dropout element-wise, then softmax row-wise.
__global__ void softmax_row_kernel(float* data, int batch_size, int out_features) {
    int row = blockIdx.x;
    if (row >= batch_size) return;

    float max_val = -1e38f;
    for (int col = 0; col < out_features; ++col) {
        max_val = fmaxf(max_val, data[row * out_features + col]);
    }
    float sum = 0.0f;
    for (int col = 0; col < out_features; ++col) {
        float val = expf(data[row * out_features + col] - max_val);
        data[row * out_features + col] = val;
        sum += val;
    }
    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int col = 0; col < out_features; ++col) {
        data[row * out_features + col] *= inv_sum;
    }
}

torch::Tensor fused_op_cuda(torch::Tensor x, float p, bool training) {
    auto out = x.clone();
    int batch_size = out.size(0);
    int out_features = out.size(1);
    float scale = 1.0f / (1.0f - p);

    if (training) {
        int total = batch_size * out_features;
        int block = 256;
        int grid = (total + block - 1) / block;
        fused_dropout_softmax_kernel<<<grid, block>>>(out.data_ptr<float>(), p, scale, batch_size, out_features, 1234ULL);
    }
    
    softmax_row_kernel<<<batch_size, 1>>>(out.data_ptr<float>(), batch_size, out_features);
    return out;
}
"""

fused_cpp_source = "torch::Tensor fused_op_cuda(torch::Tensor x, float p, bool training);"

fused_lib = load_inline(
    name="fused_lib",
    cpp_sources=fused_cpp_source,
    cuda_sources=fused_source,
    functions=["fused_op_cuda"],
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout_p = dropout_p

    def forward(self, x):
        # Matmul is highly optimized in cuBLAS, keep it as is
        x = self.matmul(x)
        # Use fused CUDA kernel for Dropout + Softmax
        return fused_lib.fused_op_cuda(x, self.dropout_p, self.training)