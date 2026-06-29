import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

maxpool1d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

template <typename scalar_t>
__global__ void maxpool1d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t* __restrict__ indices,
    int64_t batch_size,
    int64_t num_features,
    int64_t input_length,
    int64_t output_length,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = batch_size * num_features * output_length;
    
    if (idx >= total_elements) return;
    
    int64_t out_pos = idx % output_length;
    int64_t feature_idx = (idx / output_length) % num_features;
    int64_t batch_idx = idx / (num_features * output_length);
    
    int64_t padded_start = out_pos * stride - padding;
    
    scalar_t max_val = -FLT_MAX;
    int64_t max_idx = -1;
    
    for (int64_t k = 0; k < kernel_size; ++k) {
        int64_t input_pos = padded_start + k * dilation;
        
        if (input_pos >= 0 && input_pos < input_length) {
            scalar_t val = input[batch_idx * num_features * input_length + 
                                feature_idx * input_length + input_pos];
            if (val > max_val) {
                max_val = val;
                max_idx = input_pos;
            }
        }
    }
    
    output[batch_idx * num_features * output_length + 
           feature_idx * output_length + out_pos] = max_val;
    
    if (return_indices && indices != nullptr) {
        indices[batch_idx * num_features * output_length + 
                feature_idx * output_length + out_pos] = max_idx;
    }
}

torch::Tensor maxpool1d_cuda(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool return_indices) {
    
    int64_t batch_size = input.size(0);
    int64_t num_features = input.size(1);
    int64_t input_length = input.size(2);
    
    int64_t output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    if (output_length < 0) output_length = 0;
    
    auto output = torch::empty({batch_size, num_features, output_length}, input.options());
    
    torch::Tensor indices;
    if (return_indices) {
        indices = torch::empty({batch_size, num_features, output_length}, 
                               input.options().dtype(torch::kInt64));
    }
    
    const int block_size = 256;
    int64_t total_elements = batch_size * num_features * output_length;
    const int num_blocks = (total_elements + block_size - 1) / block_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "maxpool1d_cuda", ([&] {
        maxpool1d_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            return_indices ? indices.data_ptr<int64_t>() : nullptr,
            batch_size,
            num_features,
            input_length,
            output_length,
            kernel_size,
            stride,
            padding,
            dilation,
            return_indices
        );
    }));
    
    if (return_indices) {
        return std::make_tuple(output, indices);
    } else {
        return output;
    }
}
"""

maxpool1d_cpp_source = (
    "torch::Tensor maxpool1d_cuda("
    "    torch::Tensor input,"
    "    int64_t kernel_size,"
    "    int64_t stride,"
    "    int64_t padding,"
    "    int64_t dilation,"
    "    bool return_indices);"
)

maxpool1d = load_inline(
    name="maxpool1d",
    cpp_sources=maxpool1d_cpp_source,
    cuda_sources=maxpool1d_cuda_source,
    functions=["maxpool1d_cuda"],
    verbose=False,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.maxpool1d_cuda = maxpool1d.maxpool1d_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.return_indices:
            output, indices = self.maxpool1d_cuda(
                x, self.kernel_size, self.stride, self.padding, self.dilation, True
            )
            return output, indices
        else:
            return self.maxpool1d_cuda(
                x, self.kernel_size, self.stride, self.padding, self.dilation, False
            )