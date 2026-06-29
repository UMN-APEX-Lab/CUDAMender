import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Fused CUDA kernel for: add + layer_norm + avg_pool + gelu
fused_ops_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_add_norm_pool_gelu_kernel(
    const float* input,
    const float sum_weight,
    const float* gamma,
    const float* beta,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    int pool_d,
    int pool_h,
    int pool_w,
    float* output
) {
    // Each thread block handles one (b, c) channel
    int bc = blockIdx.x;
    int b = bc / channels;
    int c = bc % channels;
    
    int spatial_size = depth * height * width;
    int pool_out_d = depth / pool_d;
    int pool_out_h = height / pool_h;
    int pool_out_w = width / pool_w;
    int pool_out_spatial = pool_out_d * pool_out_h * pool_out_w;
    
    const float* in_ptr = input + bc * spatial_size;
    
    // First pass: compute mean and variance for layer norm
    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;
    
    for (int i = threadIdx.x; i < spatial_size; i += blockDim.x) {
        float val = in_ptr[i] + sum_weight;
        thread_sum += val;
        thread_sq_sum += val * val;
    }
    
    // Block-level reduction using shared memory
    __shared__ float shared_sum[256];
    __shared__ float shared_sq_sum[256];
    
    shared_sum[threadIdx.x] = thread_sum;
    shared_sq_sum[threadIdx.x] = thread_sq_sum;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            shared_sq_sum[threadIdx.x] += shared_sq_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    float mean = shared_sum[0] / spatial_size;
    float mean_sq = shared_sq_sum[0] / spatial_size;
    float var = mean_sq - mean * mean;
    float rstd = 1.0f / sqrtf(var + 1e-5f);
    
    __syncthreads();
    
    // Second pass: layer norm + average pooling + GELU
    for (int idx = threadIdx.x; idx < pool_out_spatial; idx += blockDim.x) {
        int pd = idx / (pool_out_h * pool_out_w);
        int phw = idx % (pool_out_h * pool_out_w);
        int ph = phw / pool_out_w;
        int pw = phw % pool_out_w;
        
        // Average pooling: compute mean over pool_d x pool_h x pool_w region
        float pool_sum = 0.0f;
        for (int kd = 0; kd < pool_d; kd++) {
            for (int kh = 0; kh < pool_h; kh++) {
                for (int kw = 0; kw < pool_w; kw++) {
                    int d = pd * pool_d + kd;
                    int h = ph * pool_h + kh;
                    int w = pw * pool_w + kw;
                    int in_idx = ((d * height) + h) * width + w;
                    float val = in_ptr[in_idx] + sum_weight;
                    // Layer norm
                    val = (val - mean) * rstd;
                    val = val * gamma[c] + beta[c];
                    pool_sum += val;
                }
            }
        }
        
        float pool_val = pool_sum / (pool_d * pool_h * pool_w);
        
        // GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x = pool_val;
        const float sqrt_2_over_pi = 0.7978845608028654f;
        float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
        float gelu_out = x * cdf;
        
        int out_idx = ((bc * pool_out_d + pd) * pool_out_h + ph) * pool_out_w + pw;
        output[out_idx] = gelu_out;
    }
}

torch::Tensor fused_add_norm_pool_gelu_cuda(
    torch::Tensor input,
    float sum_weight,
    torch::Tensor gamma,
    torch::Tensor beta,
    int pool_d,
    int pool_h,
    int pool_w
) {
    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);
    
    int pool_out_d = depth / pool_d;
    int pool_out_h = height / pool_h;
    int pool_out_w = width / pool_w;
    
    auto output = torch::zeros({batch_size, channels, pool_out_d, pool_out_h, pool_out_w}, input.options());
    
    int num_blocks = batch_size * channels;
    const int threads = 256;
    
    fused_add_norm_pool_gelu_kernel<<<num_blocks, threads>>>(
        input.data_ptr<float>(),
        sum_weight,
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        pool_d,
        pool_h,
        pool_w,
        output.data_ptr<float>()
    );
    
    return output;
}
"""

fused_ops_cpp_source = """
torch::Tensor fused_add_norm_pool_gelu_cuda(
    torch::Tensor input,
    float sum_weight,
    torch::Tensor gamma,
    torch::Tensor beta,
    int pool_d,
    int pool_h,
    int pool_w
);
"""

# Compile the inline CUDA code
fused_ops = load_inline(
    name="fused_ops",
    cpp_sources=fused_ops_cpp_source,
    cuda_sources=fused_ops_source,
    functions=["fused_add_norm_pool_gelu_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized Model that uses custom CUDA kernels for fused operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        self.pool_kernel_size = pool_kernel_size
        self.fused_ops = fused_ops
        
        # Pre-allocate buffers for gamma and beta
        self.register_buffer('gamma_buf', torch.ones(norm_shape[0]))
        self.register_buffer('beta_buf', torch.zeros(norm_shape[0]))

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Get gamma and beta from layer norm
        if self.norm.elementwise_affine:
            gamma = self.norm.weight
            beta = self.norm.bias
        else:
            gamma = self.gamma_buf
            beta = self.beta_buf
        
        # Reshape for fused kernel: (B, C, D, H, W) -> treat C as channel dim
        B, C, D, H, W = x.shape
        pool_d, pool_h, pool_w = self.pool_kernel_size
        
        # Call fused kernel: add + layer_norm + avg_pool + gelu
        output = self.fused_ops.fused_add_norm_pool_gelu_cuda(
            x, 
            self.sum_weight.item(),
            gamma,
            beta,
            pool_d,
            pool_h,
            pool_w
        )
        
        return output