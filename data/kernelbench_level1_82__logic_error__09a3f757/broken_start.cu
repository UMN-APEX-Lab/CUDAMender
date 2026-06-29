import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Optimized depthwise convolution CUDA kernel using shared memory and proper tiling
depthwise_conv2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define MAX_KERNEL_SIZE 7

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int out_height,
    int out_width
) {
    // Shared memory for input tile
    __shared__ float input_tile[TILE_SIZE + MAX_KERNEL_SIZE - 1][TILE_SIZE + MAX_KERNEL_SIZE - 1];
    
    int batch_idx = blockIdx.z / in_channels;
    int channel_idx = blockIdx.z % in_channels;
    
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Calculate input coordinates with padding
    int in_y_start = out_y * stride - padding;
    int in_x_start = out_x * stride - padding;
    
    // Load input into shared memory with halo
    int tile_size_with_halo = TILE_SIZE + kernel_size - 1;
    
    for (int dy = threadIdx.y; dy < tile_size_with_halo; dy += TILE_SIZE) {
        for (int dx = threadIdx.x; dx < tile_size_with_halo; dx += TILE_SIZE) {
            int in_y = in_y_start + dy;
            int in_x = in_x_start + dx;
            
            float val = 0.0f;
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int in_idx = ((batch_idx * in_channels + channel_idx) * in_height + in_y) * in_width + in_x;
                val = input[in_idx];
            }
            input_tile[dy][dx] = val;
        }
    }
    
    __syncthreads();
    
    // Compute convolution only if within output bounds
    if (out_y < out_height && out_x < out_width && 
        threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE) {
        
        float sum = 0.0f;
        
        // Get weight pointer for this channel
        const float* weight_ptr = weight + channel_idx * kernel_size * kernel_size;
        
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                float w = weight_ptr[ky * kernel_size + kx];
                float inp = input_tile[threadIdx.y + ky][threadIdx.x + kx];
                sum += inp * w;
            }
        }
        
        int out_idx = ((batch_idx * in_channels + channel_idx) * out_height + out_y) * out_width + out_x;
        output[out_idx] = sum;
    }
}

__global__ void depthwise_conv2d_with_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    int out_height,
    int out_width
) {
    // Shared memory for input tile
    __shared__ float input_tile[TILE_SIZE + MAX_KERNEL_SIZE - 1][TILE_SIZE + MAX_KERNEL_SIZE - 1];
    
    int batch_idx = blockIdx.z / in_channels;
    int channel_idx = blockIdx.z % in_channels;
    
    int out_y = blockIdx.y * TILE_SIZE + threadIdx.y;
    int out_x = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Calculate input coordinates with padding
    int in_y_start = out_y * stride - padding;
    int in_x_start = out_x * stride - padding;
    
    // Load input into shared memory with halo
    int tile_size_with_halo = TILE_SIZE + kernel_size - 1;
    
    for (int dy = threadIdx.y; dy < tile_size_with_halo; dy += TILE_SIZE) {
        for (int dx = threadIdx.x; dx < tile_size_with_halo; dx += TILE_SIZE) {
            int in_y = in_y_start + dy;
            int in_x = in_x_start + dx;
            
            float val = 0.0f;
            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                int in_idx = ((batch_idx * in_channels + channel_idx) * in_height + in_y) * in_width + in_x;
                val = input[in_idx];
            }
            input_tile[dy][dx] = val;
        }
    }
    
    __syncthreads();
    
    // Compute convolution only if within output bounds
    if (out_y < out_height && out_x < out_width && 
        threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE) {
        
        float sum = bias[channel_idx];
        
        // Get weight pointer for this channel
        const float* weight_ptr = weight + channel_idx * kernel_size * kernel_size;
        
        #pragma unroll
        for (int ky = 0; ky < kernel_size; ky++) {
            #pragma unroll
            for (int kx = 0; kx < kernel_size; kx++) {
                float w = weight_ptr[ky * kernel_size + kx];
                float inp = input_tile[threadIdx.y + ky][threadIdx.x + kx];
                sum += inp * w;
            }
        }
        
        int out_idx = ((batch_idx * in_channels + channel_idx) * out_height + out_y) * out_width + out_x;
        output[out_idx] = sum;
    }
}

torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding
) {
    // Input: (batch_size, in_channels, in_height, in_width)
    // Weight: (in_channels, 1, kernel_size, kernel_size) for depthwise
    // Actually weight is (in_channels, 1, kH, kW) but we reshape to (in_channels, kH, kW)
    
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int kernel_size = weight.size(2); // Assuming square kernel
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    auto output = torch::empty({batch_size, in_channels, out_height, out_width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    
    // Reshape weight to (in_channels, kernel_size, kernel_size)
    auto weight_reshaped = weight.view({in_channels, kernel_size, kernel_size});
    
    // Grid dimensions
    int grid_x = (out_width + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (out_height + TILE_SIZE - 1) / TILE_SIZE;
    int grid_z = batch_size * in_channels;
    
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    if (bias.has_value()) {
        depthwise_conv2d_with_bias_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight_reshaped.data_ptr<float>(),
            bias.value().data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding,
            out_height,
            out_width
        );
    } else {
        depthwise_conv2d_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight_reshaped.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding,
            out_height,
            out_width
        );
    }
    
    return output;
}
"""

depthwise_conv2d_cpp_source = """
torch::Tensor depthwise_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding
);
"""

# Compile the inline CUDA code
depthwise_conv2d = load_inline(
    name="depthwise_conv2d",
    cpp_sources=depthwise_conv2d_cpp_source,
    cuda_sources=depthwise_conv2d_source,
    functions=["depthwise_conv2d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized depthwise 2D convolution using custom CUDA kernel with shared memory tiling.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights manually
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='linear')
        
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
        self.depthwise_conv2d = depthwise_conv2d
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs optimized depthwise 2D convolution using custom CUDA kernel.
        """
        if not x.is_cuda:
            x = x.cuda()
        
        # Move parameters to same device as input if needed
        if self.weight.device != x.device:
            self.weight = self.weight.to(x.device)
            if self.bias is not None:
                self.bias = self.bias.to(x.device)
        
        c10_bias = torch::nullopt;
        if (self.bias is not None) {
            c10_bias = self.bias;
        }
        
        # Use the custom CUDA kernel
        return self.depthwise_conv2d.depthwise_conv2d_cuda(
            x, 
            self.weight, 
            self.bias,
            self.stride,
            self.padding
        )