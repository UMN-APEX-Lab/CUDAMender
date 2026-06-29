#ifndef REDUCE_WARP_H
#define REDUCE_WARP_H

#include <cuda_runtime.h>

__device__ inline float reduce_warp(float v) {
    // Use warp shuffle to perform reduction within a warp
    // Minimize data spread by keeping data in registers and using shuffle
    v += __shfl_down_sync(0xFFFFFFFF, v, 16);
    v += __shfl_down_sync(0xFFFFFFFF, v, 8);
    v += __shfl_down_sync(0xFFFFFFFF, v, 4);
    v += __shfl_down_sync(0xFFFFFFFF, v, 2);
    v += __shfl_down_sync(0xFFFFFFFF, v, 1);
    return v;
}

__device__ inline float reduce_threadblock(float v, float *smem) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warpId = tid / 32;
    const unsigned int laneId = tid % 32;
    const unsigned int numWarps = blockDim.x / 32;
    
    // First, reduce within each warp
    float warpSum = reduce_warp(v);
    
    // First thread in each warp writes to shared memory
    if (laneId == 0) {
        smem[warpId] = warpSum;
    }
    __syncthreads();
    
    // Final reduction of warp sums using first warp
    float blockSum = 0.0f;
    if (warpId == 0) {
        // Load warp sums, padding with 0 if numWarps < 32
        if (laneId < numWarps) {
            blockSum = smem[laneId];
        } else {
            blockSum = 0.0f;
        }
        blockSum = reduce_warp(blockSum);
    }
    
    return blockSum;
}

#endif // REDUCE_WARP_H