import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for transposed convolution optimized for asymmetric inputs
conv_transpose2d_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Optimized transposed 2D convolution kernel using implicit gemm approach
// Optimized for asymmetric inputs (height != width)
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int out_height,
    const int out_width,
    const int groups
) {
    // Each thread computes one output element
    // Grid: (batch, out_channel_blocks, output_spatial_blocks)
    // Block: 256 threads
    
    const int out_channel_stride = out_height * out_width;
    const int in_channel_stride = in_height * in_width;
    
    // Calculate output position
    const int out_c = blockIdx.y * blockDim.y + threadIdx.y;
    const int spatial_idx = blockIdx.z * blockDim.x + threadIdx.x;
    
    if (out_c >= out_channels || spatial_idx >= out_height * out_width) return;
    
    const int out_h = spatial_idx / out_width;
    const int out_w = spatial_idx % out_width;
    
    // Calculate which input positions contribute to this output
    // For transposed conv: output position maps to input via stride
    const int h_start = (out_h + padding) / stride;
    const int w_start = (out_w + padding) / stride;
    
    // Calculate kernel position
    const int kh_start = (out_h + padding) % stride;
    const int kw_start = (out_w + padding) % stride;
    
    // Group handling
    const int group_size_out = out_channels / groups;
    const int group_size_in = in_channels / groups;
    const int g = out_c / group_size_out;
    
    float acc = 0.0f;
    
    // Iterate over valid input positions and kernel positions
    for (int kh = kh_start; kh < kernel_size; kh += stride) {
        const int in_h = h_start - (kh - kh_start) / stride;
        if (in_h < 0 || in_h >= in_height) continue;
        
        for (int kw = kw_start; kw < kernel_size; kw += stride) {
            const int in_w = w_start - (kw - kw_start) / stride;
            if (in_w < 0 || in_w >= in_width) continue;
            
            // Compute contribution from all input channels in this group
            for (int in_c = 0; in_c < group_size_in; ++in_c) {
                const int in_c_global = g * group_size_in + in_c;
                const float in_val = input[blockIdx.x * in_channels * in_channel_stride + 
                                          in_c_global * in_channel_stride + 
                                          in_h * in_width + in_w];
                
                // Weight index: [out_c, in_c, kh, kw]
                const int w_idx = out_c * (group_size_in * kernel_size * kernel_size) + 
                                 in_c * (kernel_size * kernel_size) + 
                                 kh * kernel_size + kw;
                const float w_val = weight[w_idx];
                
                acc += in_val * w_val;
            }
        }
    }
    
    const int out_idx = blockIdx.x * out_channels * out_channel_stride + 
                       out_c * out_channel_stride + 
                       out_h * out_width + out_w;
    output[out_idx] = acc;
}

// Optimized kernel using shared memory for better memory access patterns
__global__ void conv_transpose2d_optimized_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int out_height,
    const int out_width,
    const int groups
) {
    // Use 2D thread blocks for better spatial locality
    // Each block handles a tile of output spatial positions
    
    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    // Block dimensions
    const int BLOCK_H = 8;
    const int BLOCK_W = 32;
    
    const int out_h_start = by * BLOCK_H;
    const int out_w_start = (bz % ((out_width + BLOCK_W - 1) / BLOCK_W)) * BLOCK_W;
    const int out_c = bx;
    
    if (out_c >= out_channels) return;
    
    const int out_h = out_h_start + tid / BLOCK_W;
    const int out_w = out_w_start + tid % BLOCK_W;
    
    if (out_h >= out_height || out_w >= out_width) return;
    
    const int group_size_out = out_channels / groups;
    const int group_size_in = in_channels / groups;
    const int g = out_c / group_size_out;
    
    const int in_channel_stride = in_height * in_width;
    const int out_channel_stride = out_height * out_width;
    
    // Calculate input and kernel positions
    const int h_start = (out_h + padding) / stride;
    const int w_start = (out_w + padding) / stride;
    const int kh_start = (out_h + padding) % stride;
    const int kw_start = (out_w + padding) % stride;
    
    float acc = 0.0f;
    
    // Unroll inner loops for common kernel sizes
    #pragma unroll 2
    for (int kh = kh_start; kh < kernel_size; kh += stride) {
        const int in_h = h_start - (kh - kh_start) / stride;
        if (in_h < 0 || in_h >= in_height) continue;
        
        #pragma unroll 2
        for (int kw = kw_start; kw < kernel_size; kw += stride) {
            const int in_w = w_start - (kw - kw_start) / stride;
            if (in_w < 0 || in_w >= in_width) continue;
            
            // Process input channels
            const float* in_ptr = input + blockIdx.z * in_channels * in_channel_stride + 
                                 g * group_size_in * in_channel_stride +
                                 in_h * in_width + in_w;
            
            // Weight pointer for this output channel
            const float* w_ptr = weight + out_c * group_size_in * kernel_size * kernel_size +
                                kh * kernel_size + kw;
            
            for (int in_c = 0; in_c < group_size_in; ++in_c) {
                const float in_val = in_ptr[in_c * in_channel_stride];
                const float w_val = w_ptr[in_c * kernel_size * kernel_size];
                acc += in_val * w_val;
            }
        }
    }
    
    const int out_idx = blockIdx.z * out_channels * out_channel_stride + 
                       out_c * out_channel_stride + 
                       out_h * out_width + out_w;
    output[out_idx] = acc;
}

// Launcher that selects appropriate kernel configuration
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    // Calculate output dimensions
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               input.options());
    
    // Choose kernel based on problem size
    // For large asymmetric inputs, use optimized 2D blocking
    const bool use_optimized = (out_height * out_width >= 1024);
    
    if (use_optimized) {
        const int BLOCK_H = 8;
        const int BLOCK_W = 32;
        const int threads = BLOCK_H * BLOCK_W; // 256
        
        const int grid_y = (out_height + BLOCK_H - 1) / BLOCK_H;
        const int grid_z = ((out_width + BLOCK_W - 1) / BLOCK_W) * batch_size;
        
        dim3 grid(out_channels, grid_y, grid_z);
        dim3 block(threads);
        
        conv_transpose2d_optimized_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding,
            out_height,
            out_width,
            groups
        );
    } else {
        // Simple 1D blocking for smaller problems
        const int threads = 256;
        const int spatial_size = out_height * out_width;
        const int num_spatial_blocks = (spatial_size + threads - 1) / threads;
        
        dim3 grid(batch_size, out_channels, num_spatial_blocks);
        dim3 block(threads);
        
        conv_transpose2d_kernel<<<grid, block>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            kernel_size,
            stride,
            padding,
            out_height,
            out_width,
            groups
        );
    }
    
    return output;
}

// Fused conv transpose + bias kernel
__global__ void conv_transpose2d_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int kernel_size,
    const int stride,
    const int padding,
    const int out_height,
    const int out_width,
    const int groups
) {
    const int tid = threadIdx.x;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    const int BLOCK_H = 8;
    const int BLOCK_W = 32;
    
    const int out_h_start = by * BLOCK_H;
    const int out_w_start = (bz % ((out_width + BLOCK_W - 1) / BLOCK_W)) * BLOCK_W;
    const int out_c = bx;
    
    if (out_c >= out_channels) return;
    
    const int out_h = out_h_start + tid / BLOCK_W;
    const int out_w = out_w_start + tid % BLOCK_W;
    
    if (out_h >= out_height || out_w >= out_width) return;
    
    const int group_size_out = out_channels / groups;
    const int group_size_in = in_channels / groups;
    const int g = out_c / group_size_out;
    
    const int in_channel_stride = in_height * in_width;
    const int out_channel_stride = out_height * out_width;
    
    const int h_start = (out_h + padding) / stride;
    const int w_start = (out_w + padding) / stride;
    const int kh_start = (out_h + padding) % stride;
    const int kw_start = (out_w + padding) % stride;
    
    float acc = 0.0f;
    
    #pragma unroll 2
    for (int kh = kh_start; kh < kernel_size; kh += stride) {
        const int in_h = h_start - (kh - kh_start) / stride;
        if (in_h < 0 || in_h >= in_height) continue;
        
        #pragma unroll 2
        for (int kw = kw_start; kw < kernel_size; kw += stride) {
            const int in_w = w_start - (kw - kw_start) / stride;
            if (in_w < 0 || in_w >= in_width) continue;
            
            const float* in_ptr = input + blockIdx.z * in_channels * in_channel_stride + 
                                 g * group_size_in * in_channel_stride +
                                 in_h * in_width + in_w;
            
            const float* w_ptr = weight + out_c * group_size_in * kernel_size * kernel_size +
                                kh * kernel_size + kw;
            
            for (int in_c = 0; in_c < group_size_in; ++in_c) {
                const float in_val = in_ptr[in_c * in_channel_stride];
                const float w_val = w_ptr[in_c * kernel_size * kernel_size];
                acc += in_val * w_val;
            }
        }
    }
    
    acc += bias[out_c];
    
    const int out_idx = blockIdx.z * out_channels * out_channel_stride + 
                       out_c * out_channel_stride + 
                       out_h * out_width + out_w;
    output[out_idx] = acc;
}

torch::Tensor conv_transpose2d_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups
) {
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_size = weight.size(2);
    
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_size + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_size + output_padding;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, 
                               input.options());
    
    const int BLOCK_H = 8;
    const int BLOCK_W = 32;
    const int threads = BLOCK_H * BLOCK_W;
    
    const int grid_y = (out_height + BLOCK_H - 1) / BLOCK_H;
    const int grid_z = ((out_width + BLOCK_W - 1) / BLOCK_W) * batch_size;
    
    dim3 grid(out_channels, grid_y, grid_z);
    dim3 block(threads);
    
    conv_transpose2d_bias_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_size,
        stride,
        padding,
        out_height,
        out_width,
        groups
    );
    
    return output;
}
"""

conv_transpose2d_cpp_source = """
torch::Tensor conv_transpose2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int groups
);
torch::Tensor conv_transpose2d_bias_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int output_padding,
    int groups
);
"""

# Compile the inline CUDA code
conv_transpose2d_cuda = load_inline(
    name="conv_transpose2d_cuda",
    cpp_sources=conv_transpose2d_cpp_source,
    cuda_sources=conv_transpose2d_source,
    functions=["conv_transpose2d_cuda", "conv_transpose2d_bias_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class ModelNew(nn.Module):
    """
    Optimized transposed 2D convolution with custom CUDA kernel for asymmetric inputs.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, output_padding: int = 0, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Initialize weight manually to match ConvTranspose2d format
        # PyTorch ConvTranspose2d weight shape: (in_channels, out_channels//groups, kH, kW)
        # But stored as (out_channels, in_channels//groups, kH, kW) for grouped conv
        # Actually: weight shape is (in_channels, out_channels//groups, kH, kW)
        # For custom kernel, we use (out_channels, in_channels//groups, kH, kW)
        
        # Calculate weight shape: for transposed conv, weight is (in_channels, out_channels//groups, kH, kW)
        # We transpose to (out_channels, in_channels//groups, kH, kW) for our kernel
        self.weight = nn.Parameter(torch.Tensor(
            in_channels, out_channels // groups, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        
        # Transpose weight for our kernel: (out_channels, in_channels//groups, kH, kW)
        # Actually we need to reshape: original is (in_channels, out_channels//groups, kH, kW)
        # We want (out_channels, in_channels//groups, kH, kW)
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
        self.cuda_module = conv_transpose2d_cuda
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CUDA
        if not x.is_cuda:
            x = x.cuda()
        
        # Reshape weight for our kernel
        # Original: (in_channels, out_channels//groups, kH, kW)
        # Target: (out_channels, in_channels//groups, kH, kW)
        # This is a reshape: we need to permute dimensions
        
        batch_size = x.size(0)
        in_c = x.size(1)
        
        # Permute weight: (in_channels, out_channels//groups, kH, kW) -> (out_channels//groups, in_channels, kH, kW)
        # Then reshape to (out_channels, in_channels//groups, kH, kW)
        
        # Actually let's handle the weight reshape properly
        # For groups=1: weight is (in_channels, out_channels, kH, kW) -> need (out_channels, in_channels, kH, kW)
        # For groups>1: weight is (in_channels, out_channels//groups, kH, kW) -> need (out_channels, in_channels//groups, kH, kW)
        
        # Permute: (in_channels, out_channels//groups, kH, kW) -> (out_channels//groups, in_channels, kH, kW)
        # Then view as (out_channels, in_channels//groups, kH, kW) - this works when out_channels = groups * (out_channels//groups)
        
        weight_perm = self.weight.permute(1, 0, 2, 3).contiguous()
        # Now: (out_channels//groups, in_channels, kH, kW)
        # Reshape to: (out_channels, in_channels//groups, kH, kW)
        weight_reshaped = weight_perm.view(self.out_channels, self.in_channels // self.groups, 
                                          self.kernel_size, self.kernel_size)
        
        if self.bias is not None:
            return self.cuda_module.conv_transpose2d_bias_cuda(
                x, weight_reshaped, self.bias,
                self.stride, self.padding, self.output_padding, self.groups
            )
        else:
            return self.cuda_module.conv_transpose2d_cuda(
                x, weight_reshaped,
                self.stride, self.padding, self.output_padding, self.groups
            )


def get_inputs():
    batch_size = 8
    in_channels = 32
    out_channels = 32
    kernel_size = 3
    height_in = 512
    width_in = 1024
    x = torch.rand(batch_size, in_channels, height_in, width_in).cuda()
    return [x]


def get_init_inputs():
    in_channels = 32
    out_channels = 32
    kernel_size = 3
    return [in_channels, out_channels, kernel_size]