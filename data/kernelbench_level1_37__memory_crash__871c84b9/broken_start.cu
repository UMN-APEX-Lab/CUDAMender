import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for fused Frobenius norm normalization
# Fuses: compute squared sum -> sqrt -> division
frobenius_norm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void frobenius_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_elements,
    float* __restrict__ norm_squared) {
    
    // Shared memory for block-level reduction
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread computes partial sum of squares
    float local_sum = 0.0f;
    for (int i = idx; i < total_elements; i += blockDim.x * gridDim.x) {
        float val = input[i];
        local_sum += val * val;
    }
    
    // Store in shared memory
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread writes block sum to global memory
    if (tid == 0) {
        norm_squared[blockIdx.x] = shared_mem[0];
    }
}

__global__ void divide_by_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_elements,
    float norm) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = input[idx] / norm;
    }
}

__global__ void reduce_final_kernel(
    float* __restrict__ block_sums,
    int num_blocks,
    float* __restrict__ result) {
    
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x;
    
    // Load block sums into shared memory
    float local_sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_sum += block_sums[i];
    }
    shared_mem[tid] = local_sum;
    __syncthreads();
    
    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        result[0] = sqrtf(shared_mem[0]);
    }
}

torch::Tensor frobenius_norm_cuda(torch::Tensor input) {
    auto total_elements = input.numel();
    auto output = torch::zeros_like(input);
    
    // Handle empty tensor
    if (total_elements == 0) {
        return output;
    }
    
    const int threads = 256;
    const int blocks = 256; // Fixed number of blocks for first reduction
    
    // Allocate for block-level sums and final norm
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto block_sums = torch::empty({blocks}, options);
    auto norm_result = torch::empty({1}, options);
    
    // First kernel: compute partial sums per block
    size_t shared_mem_size = threads * sizeof(float);
    frobenius_norm_kernel<<<blocks, threads, shared_mem_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        block_sums.data_ptr<float>());
    
    // Second kernel: reduce block sums and compute sqrt
    reduce_final_kernel<<<1, threads, shared_mem_size>>>(
        block_sums.data_ptr<float>(),
        blocks,
        norm_result.data_ptr<float>());
    
    // Third kernel: divide by norm
    const int num_blocks_div = (total_elements + threads - 1) / threads;
    divide_by_norm_kernel<<<num_blocks_div, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        norm_result.data_ptr<float>()[0]);
    
    return output;
}
"""

frobenius_norm_cpp_source = "torch::Tensor frobenius_norm_cuda(torch::Tensor input);"

# Compile the inline CUDA code
frobenius_norm = load_inline(
    name="frobenius_norm",
    cpp_sources=frobenius_norm_cpp_source,
    cuda_sources=frobenius_norm_source,
    functions=["frobenius_norm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs Frobenius norm normalization using custom CUDA kernel.
    """
    def __init__(self):
        """
        Initializes the Frobenius norm normalization layer with custom CUDA operator.
        """
        super(ModelNew, self).__init__()
        self.frobenius_norm = frobenius_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Frobenius norm normalization to the input tensor using fused CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with Frobenius norm normalization applied, same shape as input.
        """
        return self.frobenius_norm.frobenius_norm_cuda(x)


def get_inputs():
    batch_size = 112
    features = 64
    dim1 = 512
    dim2 = 512
    x = torch.rand(batch_size, features, dim1, dim2).cuda()
    return [x]


def get_init_inputs():
    return []