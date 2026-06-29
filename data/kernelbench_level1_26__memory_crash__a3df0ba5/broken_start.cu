import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

gelu_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride) {
        float val = x[i];
        out[i] = 0.5f * val * (1.0f + erff(val * 0.7071067811865476f));
    }
}

torch::Tensor gelu_cuda(torch::Tensor x) {
    auto out = torch::empty_like(x);
    int size = x.numel();
    const int block_size = 1024;
    int num_blocks = (size + block_size - 1) / block_size;
    
    gelu_kernel<<<num_blocks, block_size>>>(x.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
"""

gelu_cpp_source = "torch::Tensor gelu_cuda(torch::Tensor x);"

gelu_op = load_inline(
    name="gelu_op",
    cpp_sources=gelu_cpp_source,
    cuda_sources=gelu_source,
    functions=["gelu_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self.gelu_op = gelu_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_op.gelu_cuda(x)