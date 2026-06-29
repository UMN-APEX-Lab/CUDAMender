#include "reduce_warp.h"
#include "kernel_wrapper.h"
#include <stdio.h>
#include <cmath>
#include <cstdlib>

__global__ void kernel(float *pOut, const float *pIn)
{
    extern __shared__ char smem_[];
    float *smem = reinterpret_cast<float *>(smem_);
    int tx      = threadIdx.x;
    float v     = pIn[tx];
    v           = reduce_warp(v);
    v           = reduce_threadblock(v, smem);
    if (threadIdx.x == 0)
    {
        pOut[0] = v;
    }
}

void launch_with_input(int blockSize, const std::vector<float>& input, float expected, const char* test_name)
{
    float *d_output, *d_input;
    float h_output = 0.0f;
    
    cudaMalloc(&d_output, sizeof(float));
    cudaMalloc(&d_input, blockSize * sizeof(float));
    cudaMemcpy(d_input, input.data(), blockSize * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(blockSize);
    dim3 numBlocks(1);

    cudaLaunchConfig_t config = {0};
    config.gridDim            = numBlocks;
    config.blockDim           = threadsPerBlock;
    config.dynamicSmemBytes   = ((config.blockDim.x + 31) / 32) * sizeof(float);
    cudaLaunchKernelEx(&config, kernel, d_output, d_input);
    
    cudaDeviceSynchronize();
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    
    // Verify result
    float diff = fabs(h_output - expected);
    float tolerance = 1e-2f;
    
    if (diff < tolerance) {
        printf("  PASS [%s]: expected=%.2f, got=%.2f\n", test_name, expected, h_output);
    } else {
        printf("  FAIL [%s]: expected=%.2f, got=%.2f (diff=%.6f)\n", test_name, expected, h_output, diff);
        exit(1);
    }
    
}