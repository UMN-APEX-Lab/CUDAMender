// file: solution.cu
#include "reduce.h"
#include <float.h>

__global__ void reduce(float *gdata, float *out, size_t n)
{
    __shared__ float sdata[BLOCK_SIZE];
    int tid    = threadIdx.x;
    sdata[tid] = -FLT_MAX;
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;

    while (idx < n)
    {
        sdata[tid] = max(gdata[idx], sdata[tid]);
        idx += gridDim.x * blockDim.x;
    }

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tid < s) sdata[tid] = max(sdata[tid + s], sdata[tid]);
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}
