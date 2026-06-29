import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for mean reduction
mean_reduction_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void mean_reduction_kernel(
    const float* input,
    float* output,
    int num_outputs,
    int reduce_size,
    int stride) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_outputs) {
        float sum = 0.0f;
        for (int i = 0; i < reduce_size; i++) {
            int input_idx = idx * reduce_size + i;
            int input_offset = input_idx * stride;
            sum += input[input_offset];
        }
        output[idx] = sum / reduce_size;
    }
}

torch::Tensor mean_reduction_cuda(
    torch::Tensor input,
    int reduce_dim,
    int num_outputs,
    int reduce_size,
    int stride) {
    
    auto output = torch::empty(num_outputs, input.options());
    
    const int block_size = 256;
    const int num_blocks = (num_outputs + block_size - 1) / block_size;
    
    mean_reduction_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_outputs,
        reduce_size,
        stride);
    
    return output;
}
"""

mean_reduction_cpp_source = (
    "torch::Tensor mean_reduction_cuda("
    "torch::Tensor input, "
    "int reduce_dim, "
    "int num_outputs, "
    "int reduce_size, "
    "int stride);"
)

# Compile the inline CUDA code for mean reduction
mean_reduction = load_inline(
    name="mean_reduction",
    cpp_sources=mean_reduction_cpp_source,
    cuda_sources=mean_reduction_source,
    functions=["mean_reduction_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized model that performs mean reduction over a specific dimension
    using custom CUDA operators.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.mean_reduction = mean_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reduces the input tensor along the specified dimension by taking the mean
        using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of arbitrary shape.

        Returns:
            torch.Tensor: Output tensor with reduced dimension.
        """
        # Ensure input is on CUDA
        if x.device.type != 'cuda':
            x = x.cuda()
        
        # Calculate dimensions for the reduction
        dim = self.dim
        if dim < 0:
            dim = x.dim() + dim
        
        # Calculate the size of the dimension to reduce
        reduce_size = x.size(dim)
        
        # Calculate num_outputs (product of all dimensions except the reduced one)
        num_outputs = 1
        for i in range(x.dim()):
            if i != dim:
                num_outputs *= x.size(i)
        
        # Calculate stride (product of dimensions after the reduced dimension)
        stride = 1
        for i in range(dim + 1, x.dim()):
            stride *= x.size(i)
        
        # Call the custom CUDA kernel
        output = self.mean_reduction.mean_reduction_cuda(
            x, dim, num_outputs, reduce_size, stride
        )
        
        # Reshape output to the expected shape
        output_shape = []
        for i in range(x.dim()):
            if i != dim:
                output_shape.append(x.size(i))
        
        return output.view(output_shape)


def get_inputs():
    batch_size = 128
    dim1 = 4096
    dim2 = 4095
    x = torch.rand(batch_size, dim1, dim2).cuda()
    return [x]


def get_init_inputs():
    return [1]