#include "lcg_kernel.h"
#include <stdint.h>

#ifndef LCG_A
#define LCG_A 1664525u
#endif

#ifndef LCG_C
#define LCG_C 1013904223u
#endif

#ifndef LCG_STEPS
#define LCG_STEPS 100
#endif

static __device__ __forceinline__ uint32_t lcg_100(uint32_t x) {
    #pragma unroll
    for (int i = 0; i < LCG_STEPS; ++i) {
        x = x * (uint32_t)LCG_A + (uint32_t)LCG_C;
    }
    return x;
}

__global__ void k_calculateLCG(uint32_t * __restrict__ data_d, int numElements) {
    extern __shared__ uint32_t smem[];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    if (gid < numElements) {
        __pipeline_memcpy_async(&smem[tid], &data_d[gid], sizeof(uint32_t));
    } else {
        smem[tid] = 0u;
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
#else
    smem[tid] = (gid < numElements) ? data_d[gid] : 0u;
    __syncthreads();
#endif

    uint32_t x = smem[tid];
    x = lcg_100(x);

    if (gid < numElements) {
        data_d[gid] = x;
    }
}