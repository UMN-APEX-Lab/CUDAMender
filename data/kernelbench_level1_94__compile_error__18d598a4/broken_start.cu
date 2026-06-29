import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_cpp_source = r"""
torch::Tensor mse_mean_cuda(torch::Tensor predictions, torch::Tensor targets);
"""

_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

__global__ void mse_partial_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* __restrict__ partials,
    int64_t n
) {
    float sum = 0.0f;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < n; i += stride) {
        float d = predictions[i] - targets[i];
        sum += d * d;
    }

    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = sum;
    }
}

__global__ void finalize_kernel(
    const float* __restrict__ partials,
    float* __restrict__ out,
    int64_t n_partials,
    int64_t n
) {
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < n_partials; i += blockDim.x) {
        sum += partials[i];
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        out[0] = sum / (float)n;
    }
}

torch::Tensor mse_mean_cuda(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.scalar_type() == torch::kFloat32, "predictions must be float32");
    TORCH_CHECK(targets.scalar_type() == torch::kFloat32, "targets must be float32");
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "predictions and targets must have the same shape");

    auto p = predictions.contiguous();
    auto t = targets.contiguous();

    int64_t n = p.numel();
    auto out = torch::empty({}, p.options());

    if (n == 0) {
        out.fill_(0);
        return out;
    }

    const int threads = 256;
    int64_t max_blocks = 65535;
    int64_t blocks64 = (n + threads - 1) / threads;
    blocks64 = blocks64 > max_blocks ? max_blocks : blocks64;
    int blocks = (int)blocks64;

    auto partials = torch::empty({blocks}, p.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    mse_partial_kernel<<<blocks, threads, 0, stream>>>(
        p.data_ptr<float>(),
        t.data_ptr<float>(),
        partials.data_ptr<float>(),
        n
    );

    finalize_kernel<<<1, threads, 0, stream>>>(
        partials.data_ptr<float>(),
        out.data_ptr<float>(),
        blocks,
        n
    );

    return out;
}
"""

_ext = None
if torch.cuda.is_available():
    _ext = load_inline(
        name="mse_mean_ext",
        cpp_sources=_cpp_source,
        cuda_sources=_cuda_source,
        functions=["mse_mean_cuda"],
        verbose=False,
    )


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._ext = _ext

    def forward(self, predictions, targets):
        if self._ext is not None and predictions.is_cuda and targets.is_cuda:
            return self._ext.mse_mean_cuda(predictions, targets)
        return torch.mean((predictions - targets) ** 2)


batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    scale = torch.rand(())
    return [torch.rand(batch_size, *input_shape) * scale, torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return []