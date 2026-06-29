#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "reduce.h"

__global__ void reduce(float *gdata, float *out, size_t n) {
    // Shared memory for the block reduction
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    // Handle cases where n is not a multiple of BLOCK_SIZE
    float local_max = -1e38f; // Initialize with a very small value
    if (i < n) {
        local_max = gdata[i];
    }
    sdata[tid] = local_max;
    __syncthreads();

    // Parallel sweep reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write the block maximum to the output array
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}