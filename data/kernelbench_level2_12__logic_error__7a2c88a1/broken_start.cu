import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Gemm + Multiply + LeakyReLU
fused_gemm_multiply_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16

// LeakyReLU kernel: out = x > 0 ? x : negative_slope * x
__global__ void multiply_leaky_relu_kernel(float* data, int size, float multiplier, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx] * multiplier;
        data[idx] = val > 0.0f ? val : negative_slope * val;
    }
}

// Fused kernel for bias add + multiply + leaky relu
__global__ void bias_multiply_leaky_relu_kernel(float* data, const float* bias, int batch_size, int out_features, float multiplier, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_features;
    if (idx < total) {
        int feat_idx = idx % out_features;
        float val = data[idx] + bias[feat_idx];
        val = val * multiplier;
        data[idx] = val > 0.0f ? val : negative_slope * val;
    }
}

torch::Tensor fused_gemm_multiply_leaky_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float multiplier, float negative_slope) {
    auto batch_size = input.size(0);
    auto in_features = input.size(1);
    auto out_features = weight.size(0);
    
    // Create output tensor
    auto output = torch::empty({batch_size, out_features}, input.options());
    
    // cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Perform matrix multiplication: output = input * weight^T
    // input: [batch_size, in_features], weight: [out_features, in_features]
    // We need output = input * weight^T, so we use C = alpha * op(A) * op(B) + beta * C
    // Here: A = input [batch_size, in_features], B = weight [out_features, in_features]
    // C = output [batch_size, out_features]
    // op(A) = A (no transpose), op(B) = B^T (transpose)
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_features, batch_size, in_features,
                &alpha,
                weight.data_ptr<float>(), in_features,
                input.data_ptr<float>(), in_features,
                &beta,
                output.data_ptr<float>(), out_features);
    
    cublasDestroy(handle);
    
    // Fused bias + multiply + leaky relu
    int total_elements = batch_size * out_features;
    const int block_size = 256;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    bias_multiply_leaky_relu_kernel<<<num_blocks, block_size>>>(
        output.data_ptr<float>(),
        bias.data_ptr<float>(),
        batch_size,
        out_features,
        multiplier,
        negative_slope
    );
    
    return output;
}

// Simple kernel for gemm without bias (used when bias is None, though we always have bias here)
__global__ void multiply_leaky_relu_no_bias_kernel(float* data, int size, float multiplier, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx] * multiplier;
        data[idx] = val > 0.0f ? val : negative_slope * val;
    }
}
"""

fused_gemm_multiply_leaky_relu_cpp_source = (
    "torch::Tensor fused_gemm_multiply_leaky_relu_cuda(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float multiplier, float negative_slope);"
)

# Compile the inline CUDA code
fused_gemm_multiply_leaky_relu = load_inline(
    name="fused_gemm_multiply_leaky_relu",
    cpp_sources=fused_gemm_multiply_leaky_relu_cpp_source,
    cuda_sources=fused_gemm_multiply_leaky_relu_source,
    functions=["fused_gemm_multiply_leaky_relu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublas"],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs a fused Gemm + Multiply + LeakyReLU using custom CUDA kernel.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multiplier = multiplier
        self.negative_slope = negative_slope
        
        # Initialize weight and bias manually (transposed for cuBLAS)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=negative_slope)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_gemm_multiply_leaky_relu

    def forward(self, x):
        return self.fused_op.fused_gemm_multiply_leaky_relu_cuda(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
    )


def get_inputs():
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]