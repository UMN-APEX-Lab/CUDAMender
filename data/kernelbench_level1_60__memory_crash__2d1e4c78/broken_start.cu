import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for 3D convolution using im2col + GEMM approach
# This fuses the im2col transformation with the matrix multiplication for better performance
conv3d_cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>

__global__ void im2col_3d_kernel(const float* input, float* col, int batch_size,
                                  int in_c, int in_d, int in_h, int in_w,
                                  int kd, int kh, int kw,
                                  int stride_d, int stride_h, int stride_w,
                                  int pad_d, int pad_h, int pad_w,
                                  int dil_d, int dil_h, int dil_w,
                                  int out_d, int out_h, int out_w) {
    int total = batch_size * out_d * out_h * out_w * in_c * kd * kh * kw;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        // Decode index
        int rem = i;
        int batch = rem / (out_d * out_h * out_w * in_c * kd * kh * kw);
        rem %= (out_d * out_h * out_w * in_c * kd * kh * kw);
        int od = rem / (out_h * out_w * in_c * kd * kh * kw);
        rem %= (out_h * out_w * in_c * kd * kh * kw);
        int oh = rem / (out_w * in_c * kd * kh * kw);
        rem %= (out_w * in_c * kd * kh * kw);
        int ow = rem / (in_c * kd * kh * kw);
        rem %= (in_c * kd * kh * kw);
        int c = rem / (kd * kh * kw);
        rem %= (kd * kh * kw);
        int dk = rem / (kh * kw);
        rem %= (kh * kw);
        int hk = rem / kw;
        int wk = rem % kw;
        
        // Compute input position
        int id = od * stride_d + dk * dil_d - pad_d;
        int ih = oh * stride_h + hk * dil_h - pad_h;
        int iw = ow * stride_w + wk * dil_w - pad_w;
        
        int col_idx = ((batch * out_d + od) * out_h + oh) * out_w + ow;
        col_idx = col_idx * (in_c * kd * kh * kw) + c * (kd * kh * kw) + dk * (kh * kw) + hk * kw + wk;
        
        if (id >= 0 && id < in_d && ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            int in_idx = ((batch * in_c + c) * in_d + id) * in_h + ih;
            in_idx = in_idx * in_w + iw;
            col[col_idx] = input[in_idx];
        } else {
            col[col_idx] = 0.0f;
        }
    }
}

__global__ void gemm_kernel(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    // A: M x K (weight), B: K x N (col), C: M x N (output)
    // Each thread computes one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int m = i / N;
        int n = i % N;
        
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[m * K + k] * B[k * N + n];
        }
        C[m * N + n] = sum;
    }
}

__global__ void add_bias_kernel(float* output, const float* bias,
                                int batch_size, int out_c, int out_d, int out_h, int out_w) {
    int total = batch_size * out_c * out_d * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        // Decode: batch, c, d, h, w
        int rem = i;
        int batch = rem / (out_c * out_d * out_h * out_w);
        rem %= (out_c * out_d * out_h * out_w);
        int c = rem / (out_d * out_h * out_w);
        rem %= (out_d * out_h * out_w);
        int d = rem / (out_h * out_w);
        rem %= (out_h * out_w);
        int h = rem / out_w;
        int w = rem % out_w;
        
        int out_idx = ((batch * out_c + c) * out_d + d) * out_h + h;
        out_idx = out_idx * out_w + w;
        output[out_idx] += bias[c];
    }
}

torch::Tensor conv3d_cuda_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                                  int stride_d, int stride_h, int stride_w,
                                  int pad_d, int pad_h, int pad_w,
                                  int dil_d, int dil_h, int dil_w,
                                  int groups) {
    // input: [batch, in_c, in_d, in_h, in_w]
    // weight: [out_c, in_c/groups, kd, kh, kw]
    int batch_size = input.size(0);
    int in_c = input.size(1);
    int in_d = input.size(2);
    int in_h = input.size(3);
    int in_w = input.size(4);
    
    int out_c = weight.size(0);
    int kernel_d = weight.size(2);
    int kernel_h = weight.size(3);
    int kernel_w = weight.size(4);
    
    int out_d = (in_d + 2 * pad_d - dil_d * (kernel_d - 1) - 1) / stride_d + 1;
    int out_h = (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    // Flatten weight: [out_c, in_c/groups * kd * kh * kw]
    int kernel_size = in_c / groups * kernel_d * kernel_h * kernel_w;
    auto weight_flat = weight.reshape({out_c, kernel_size});
    
    // Output tensor
    auto output = torch::zeros({batch_size, out_c, out_d, out_h, out_w}, input.options());
    
    if (groups == 1) {
        // Standard convolution: im2col + GEMM
        // col: [batch * out_d * out_h * out_w, in_c * kd * kh * kw]
        int col_rows = batch_size * out_d * out_h * out_w;
        int col_cols = in_c * kernel_d * kernel_h * kernel_w;
        auto col = torch::zeros({col_rows, col_cols}, input.options());
        
        const int threads = 256;
        const int blocks_im2col = (col_rows * col_cols + threads - 1) / threads;
        
        im2col_3d_kernel<<<blocks_im2col, threads>>>(
            input.data_ptr<float>(), col.data_ptr<float>(),
            batch_size, in_c, in_d, in_h, in_w,
            kernel_d, kernel_h, kernel_w,
            stride_d, stride_h, stride_w,
            pad_d, pad_h, pad_w,
            dil_d, dil_h, dil_w,
            out_d, out_h, out_w);
        
        // GEMM: output = weight * col^T, but we do col * weight^T for layout
        // Actually: [out_c, kernel_size] x [kernel_size, col_rows] = [out_c, col_rows]
        // Then reshape to [batch, out_c, out_d, out_h, out_w]
        auto gemm_out = torch::mm(weight_flat, col.t()); // [out_c, col_rows]
        
        // Reshape and permute to get correct output layout
        output = gemm_out.reshape({out_c, batch_size, out_d, out_h, out_w}).permute({1, 0, 2, 3, 4}).contiguous();
        
    } else {
        // Grouped convolution - process each group separately
        int in_c_per_group = in_c / groups;
        int out_c_per_group = out_c / groups;
        
        for (int g = 0; g < groups; g++) {
            auto input_g = input.slice(1, g * in_c_per_group, (g + 1) * in_c_per_group);
            auto weight_g = weight.slice(0, g * out_c_per_group, (g + 1) * out_c_per_group);
            
            int col_rows = batch_size * out_d * out_h * out_w;
            int col_cols = in_c_per_group * kernel_d * kernel_h * kernel_w;
            auto col = torch::zeros({col_rows, col_cols}, input.options());
            
            const int threads = 256;
            const int blocks_im2col = (col_rows * col_cols + threads - 1) / threads;
            
            im2col_3d_kernel<<<blocks_im2col, threads>>>(
                input_g.data_ptr<float>(), col.data_ptr<float>(),
                batch_size, in_c_per_group, in_d, in_h, in_w,
                kernel_d, kernel_h, kernel_w,
                stride_d, stride_h, stride_w,
                pad_d, pad_h, pad_w,
                dil_d, dil_h, dil_w,
                out_d, out_h, out_w);
            
            auto weight_g_flat = weight_g.reshape({out_c_per_group, col_cols});
            auto gemm_out = torch::mm(weight_g_flat, col.t());
            
            auto out_g = gemm_out.reshape({out_c_per_group, batch_size, out_d, out_h, out_w}).permute({1, 0, 2, 3, 4});
            output.slice(1, g * out_c_per_group, (g + 1) * out_c_per_group) = out_g;
        }
    }
    
    // Add bias if present
    if (bias.defined() && bias.numel() > 0) {
        const int threads = 256;
        const int blocks_bias = (batch_size * out_c * out_d * out_h * out_w + threads - 1) / threads;
        add_bias_kernel<<<blocks_bias, threads>>>(
            output.data_ptr<float>(), bias.data_ptr<float>(),
            batch_size, out_c, out_d, out_h, out_w);
    }
    
    return output;
}

torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias,
                          std::vector<int64_t> stride, std::vector<int64_t> padding, 
                          std::vector<int64_t> dilation, int64_t groups) {
    // Handle different stride/padding/dilation formats
    int stride_d, stride_h, stride_w;
    if (stride.size() == 1) {
        stride_d = stride_h = stride_w = stride[0];
    } else {
        stride_d = stride[0];
        stride_h = stride[1];
        stride_w = stride[2];
    }
    
    int pad_d, pad_h, pad_w;
    if (padding.size() == 1) {
        pad_d = pad_h = pad_w = padding[0];
    } else {
        pad_d = padding[0];
        pad_h = padding[1];
        pad_w = padding[2];
    }
    
    int dil_d, dil_h, dil_w;
    if (dilation.size() == 1) {
        dil_d = dil_h = dil_w = dilation[0];
    } else {
        dil_d = dilation[0];
        dil_h = dilation[1];
        dil_w = dilation[2];
    }
    
    torch::Tensor bias_tensor = bias.has_value() ? bias.value() : torch::Tensor();
    
    return conv3d_cuda_forward(input, weight, bias_tensor,
                               stride_d, stride_h, stride_w,
                               pad_d, pad_h, pad_w,
                               dil_d, dil_h, dil_w,
                               groups);
}
"""

conv3d_cuda_cpp_source = """
torch::Tensor conv3d_cuda(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias,
                          std::vector<int64_t> stride, std::vector<int64_t> padding, 
                          std::vector<int64_t> dilation, int64_t groups);
"""

# Compile the inline CUDA code
conv3d_cuda = load_inline(
    name="conv3d_cuda",
    cpp_sources=conv3d_cuda_cpp_source,
    cuda_sources=conv3d_cuda_source,
    functions=["conv3d_cuda"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=[""],
)


class Conv3dCustom(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=False):
        super(Conv3dCustom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, 
                                               *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        
    def forward(self, x):
        # Convert stride, padding, dilation to tuples if needed
        if isinstance(self.stride, int):
            stride = (self.stride,) * 3
        else:
            stride = self.stride if isinstance(self.stride, (list, tuple)) else (self.stride,)
            if len(stride) == 1:
                stride = (stride[0],) * 3
            elif len(stride) == 3:
                stride = tuple(stride)
            else:
                stride = (stride[0], stride[1], stride[2])
        
        if isinstance(self.padding, int):
            padding = (self.padding,) * 3
        else:
            padding = self.padding if isinstance(self.padding, (list, tuple)) else (self.padding,)
            if len(padding) == 1:
                padding = (padding[0],) * 3
            elif len(padding) == 3:
                padding = tuple(padding)
            else:
                padding = (padding[0], padding[1], padding[2])
        
        if isinstance(self.dilation, int):
            dilation = (self.dilation,) * 3
        else:
            dilation = self.dilation if isinstance(self.dilation, (list, tuple)) else (self.dilation,)
            if len(dilation) == 1:
                dilation = (dilation[0],) * 3
            elif len(dilation) == 3:
                dilation = tuple(dilation)
            else:
                dilation = (dilation[0], dilation[1], dilation[2])
        
        return conv3d_cuda.conv3d_cuda(x, self.weight, self.bias, 
                                       list(stride), list(padding), list(dilation), self.groups)


class ModelNew(nn.Module):
    """
    Optimized 3D convolution using custom CUDA kernel with im2col + GEMM approach.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.conv3d = Conv3dCustom(in_channels, out_channels, kernel_size, 
                                   stride=stride, padding=padding, 
                                   dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3d(x)


def get_inputs():
    batch_size = 16
    in_channels = 3
    width = 64
    height = 64
    depth = 64
    x = torch.rand(batch_size, in_channels, width, height, depth).cuda()
    return [x]


def get_init_inputs():
    in_channels = 3
    out_channels = 64
    kernel_size = (3, 5, 7)
    return [in_channels, out_channels, kernel_size]