import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for fused Gemm + Multiply + LeakyReLU
fused_gemm_multiply_leaky_relu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16

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
    
    // PyTorch uses row-major storage. cuBLAS uses column-major.
    // To compute: output = input @ weight.T
    // input: [batch_size, in_features]
    // weight: [out_features, in_features]
    // output: [batch_size, out_features]
    
    // In column-major terms:
    // input_colmajor = [in_features, batch_size] with leading dim in_features
    // weight_colmajor = [in_features, out_features] with leading dim in_features  
    // output_colmajor = [out_features, batch_size] with leading dim out_features
    
    // We want: output[row,col] = sum_k input[row,k] * weight[col,k]
    // In column-major: output^T[col,row] = sum_k weight^T[k,col] * input^T[k,row]
    // So: C = B^T * A where A=input_colmajor, B=weight_colmajor
    // C = alpha * op(B) * op(A) with op(B)=T, op(A)=N
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // m = rows of C = out_features
    // n = cols of C = batch_size
    // k = inner dim = in_features
    
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,  // op(B)=N, op(A)=T
                out_features, batch_size, in_features,  // m, n, k
                &alpha,
                weight.data_ptr<float>(), in_features,  // B, ldb (in_features)
                input.data_ptr<float>(), in_features,   // A, lda (in_features)
                &beta,
                output.data_ptr<float>(), out_features);  // C, ldc (out_features)
    
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
        
        # Initialize weight and bias manually
        # Weight shape: [out_features, in_features] - same as nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize using Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_gemm_multiply_leaky_relu

    def forward(self, x):
        return self.fused_op.fused_gemm_multiply_leaky_relu_cuda(
            x, self.weight, self.bias, self.multiplier, self.negative_slope
    )


def get_inputs():
    batch_size = 1024
    in_features = 8192
    return [torch.rand(batch_size, in_features)]

def get_init_inputs():
    in_features = 8192
    out_features = 8192
    multiplier = 2.0
    negative_slope = 0.1
    return [in_features, out_features, multiplier, negative_slope]