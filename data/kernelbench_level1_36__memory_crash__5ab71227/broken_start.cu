import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void rmsnorm_kernel(const float* x, float* out, int64_t num_rows, int features, float eps, 
                               int64_t stride0, int64_t stride1, int64_t stride2, int64_t D1, int64_t D2) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    int64_t d2 = idx % D2;
    int64_t d1 = (idx / D2) % D1;
    int64_t b = idx / (D1 * D2);

    const float* x_ptr = x + b * stride0 + d1 * stride1 + d2 * stride2;
    float* out_ptr = out + b * stride0 + d1 * stride1 + d2 * stride2;

    float sum_sq = 0.0f;
    for (int i = 0; i < features; ++i) {
        float val = x_ptr[i * stride1];
        sum_sq += val * val;
    }
    float rms = sqrtf(sum_sq / features + eps);
    for (int i = 0; i < features; ++i) {
        out_ptr[i * stride1] = x_ptr[i * stride1] / rms;
    }
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    int features = x.size(1);
    int64_t D1 = x.size(2);
    int64_t D2 = x.size(3);
    int64_t num_rows = x.size(0) * D1 * D2;
    int64_t stride0 = x.stride(0);
    int64_t stride1 = x.stride(1);
    int64_t stride2 = x.stride(2);
    
    const int block_size = 256;
    const int num_blocks = (num_rows + block_size - 1) / block_size;

    rmsnorm_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), num_rows, features, eps,
        stride0, stride1, stride2, D1, D2
    );
    
    return out;
}
"""

cpp_source = "torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps);"

rmsnorm_op = load_inline(
    name="rmsnorm_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["rmsnorm_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
)

class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.rmsnorm = rmsnorm_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm.rmsnorm_cuda(x, self.eps)