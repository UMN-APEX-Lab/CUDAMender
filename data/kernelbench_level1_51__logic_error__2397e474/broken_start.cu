import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for argmax
argmax_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

template<typename scalar_t>
__global__ void argmax_kernel(const scalar_t* input, int64_t* output, 
                              int n_elements, int reduce_size, int inner_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_elements) return;
    
    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;
    
    scalar_t max_val = -FLT_MAX;
    int max_idx = 0;
    
    for (int i = 0; i < reduce_size; i++) {
        int input_idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        scalar_t val = input[input_idx];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    
    output[idx] = max_idx;
}

torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
    auto sizes = input.sizes();
    int ndim = sizes.size();
    
    // Calculate dimensions
    int reduce_size = 1;
    int outer_size = 1;
    int inner_size = 1;
    
    if (dim < 0) dim = ndim + dim;
    
    // Calculate the size of the dimension to reduce over
    reduce_size = sizes[dim];
    
    // Calculate outer_size (elements before reduce dim)
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    
    // Calculate inner_size (elements after reduce dim)
    for (int i = dim + 1; i < ndim; i++) {
        inner_size *= sizes[i];
    }
    
    // Output shape: remove the reduce dimension
    int n_elements = outer_size * inner_size;
    
    auto output = torch::empty({n_elements}, torch::TensorOptions().dtype(torch::kInt64).device(input.device()));
    
    const int block_size = 256;
    const int num_blocks = (n_elements + block_size - 1) / block_size;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_cuda", ([&] {
        argmax_kernel<scalar_t><<<num_blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            n_elements,
            reduce_size,
            inner_size
        );
    }));
    
    return output;
}
"""

argmax_cpp_source = (
    "torch::Tensor argmax_cuda(torch::Tensor input, int dim);"
)

# Compile the inline CUDA code for argmax
argmax = load_inline(
    name="argmax_custom",
    cpp_sources=argmax_cpp_source,
    cuda_sources=argmax_source,
    functions=["argmax_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to perform argmax.

        Args:
            dim (int): The dimension to perform argmax over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.argmax_op = argmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        return self.argmax_op.argmax_cuda(x, self.dim)