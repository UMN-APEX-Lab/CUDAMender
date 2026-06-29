import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel for A.T @ B matrix multiplication using shared memory tiling
# This computes C = A^T @ B where A is (K, M) and B is (K, N)
# Result C is (M, N)

matmul_transpose_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_M 64
#define TILE_N 64
#define TILE_K 16

__global__ void matmul_transpose_kernel(
    const float* __restrict__ A,  // (K, M)
    const float* __restrict__ B,  // (K, N)
    float* __restrict__ C,        // (M, N)
    int M, int N, int K
) {
    // Block indices
    int block_m = blockIdx.y;  // row in C (M dimension)
    int block_n = blockIdx.x;  // col in C (N dimension)
    
    // Thread indices within block
    int thread_m = threadIdx.y;  // 0 to TILE_M-1
    int thread_n = threadIdx.x;  // 0 to TILE_N-1
    
    // Global position in C
    int m = block_m * TILE_M + thread_m;
    int n = block_n * TILE_N + thread_n;
    
    // Accumulator
    float sum = 0.0f;
    
    // Shared memory for tiles of A^T and B
    // A^T tile: we read A as (K, M) but need A^T which is (M, K)
    // So we load A[k, m] into shared memory
    __shared__ float sA[TILE_K][TILE_M];  // Transposed: stored as (TILE_K, TILE_M)
    __shared__ float sB[TILE_K][TILE_N];  // B is (K, N), so tile is (TILE_K, TILE_N)
    
    // Loop over K dimension in tiles
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // Load A tile: A is (K, M), we need A[k, m] for k in [k_tile, k_tile+TILE_K)
        // A[k, m] where m = block_m * TILE_M + thread_m, but we need to handle the transpose
        // Actually A is (K, M), A^T is (M, K), so (A^T)[m, k] = A[k, m]
        
        // Load sA: each thread loads one or more elements
        // We want sA[kk][mm] = A[k_tile + kk][block_m * TILE_M + mm]
        // Load transpose: thread (thread_m, thread_n) helps load
        int load_k = k_tile + thread_n;  // use thread_n for K dimension
        int load_m = block_m * TILE_M + thread_m;
        
        // Collaborative loading of A tile
        // Each thread loads one element of A into sA
        if (load_k < K && load_m < M) {
            sA[thread_n][thread_m] = A[load_k * M + load_m];
        } else {
            sA[thread_n][thread_m] = 0.0f;
        }
        
        // Load B tile: B is (K, N)
        // sB[kk][nn] = B[k_tile + kk][block_n * TILE_N + nn]
        int load_n = block_n * TILE_N + thread_n;
        int load_k_b = k_tile + thread_m;  // use thread_m for K dimension
        
        if (load_k_b < K && load_n < N) {
            sB[thread_m][thread_n] = B[load_k_b * N + load_n];
        } else {
            sB[thread_m][thread_n] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk++) {
            sum += sA[kk][thread_m] * sB[kk][thread_n];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (m < M && n < N) {
        C[m * N + n] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // A: (K, M), B: (K, N)
    // Output: (M, N)
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    matmul_transpose_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );
    
    return C;
}
"""

matmul_transpose_cpp_source = "torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the inline CUDA code
matmul_transpose = load_inline(
    name="matmul_transpose",
    cpp_sources=matmul_transpose_cpp_source,
    cuda_sources=matmul_transpose_source,
    functions=["matmul_transpose_cuda"],
    verbose=True,
    extra_cflags=["-O3", "--use_fast_math"],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication (C = A.T @ B) using custom CUDA kernel
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.matmul_transpose = matmul_transpose
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication C = A.T @ B using optimized CUDA kernel.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Ensure inputs are contiguous and on CUDA
        if not A.is_cuda:
            A = A.cuda()
        if not B.is_cuda:
            B = B.cuda()
        
        # Call custom CUDA kernel
        return self.matmul_transpose.matmul_transpose_cuda(A, B)

def get_inputs():
    A = torch.rand(8192, 2048, dtype=torch.float32)
    B = torch.rand(8192, 4096, dtype=torch.float32)
    return [A, B]

def get_init_inputs():
    return []