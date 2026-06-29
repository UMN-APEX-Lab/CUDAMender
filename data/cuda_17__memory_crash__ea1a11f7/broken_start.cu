#include "reduce.h"
#include <cuda_runtime.h>
#include <algorithm>

__global__ void reduce(float *gdata, float *out, size_t n) {
    // Shared memory for the block reduction
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    // If the index is out of bounds, initialize with a very small number
    float val = -1e38f; // Approximation of -FLT_MAX
    if (i < n) {
        val = gdata[i];
    }
    sdata[tid] = val;
    __syncthreads();

    // Parallel sweep reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // The first thread of each block writes the block maximum to the output array
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}
