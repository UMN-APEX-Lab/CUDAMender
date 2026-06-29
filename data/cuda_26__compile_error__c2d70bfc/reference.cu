#include "reduce_warp.h"

__device__ float reduce_warp(float v)
{
    const unsigned MASK = 0xFFFFFFFF;
    v                   = v + __shfl_down_sync(MASK, v, 16);
    v                   = v + __shfl_down_sync(MASK, v, 8);
    v                   = v + __shfl_down_sync(MASK, v, 4);
    v                   = v + __shfl_down_sync(MASK, v, 2);
    v                   = v + __shfl_down_sync(MASK, v, 1);
    return v;
}

__device__ float reduce_threadblock(float v, float *smem)
{
    // here we assume that reduce_warp is already used
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    if (laneId == 0)
    {
        smem[warpId] = v;
    }
    // make sure every warp has written to shared memory

    // also make sure that they don't start executing before all
    // of them reach the same execution point
    __syncthreads();
    // each thread block can be of maximum size of 1024 threads
    // which means, there can be at most 32 warps in threadblock
    // we use the first warp to do the final reduction
    if (warpId == 0)
    {
        // Calculate the number of warps in the threadblock
        int numWarps = (blockDim.x + 31) / 32;
        // Only read valid data from smem, initialize rest to 0
        v = (laneId < numWarps) ? smem[laneId] : 0.0f;
        v = reduce_warp(v);
    }
    return v;
}