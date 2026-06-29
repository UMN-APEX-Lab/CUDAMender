#include "angular_momentum.h"

__global__ void k_computeAngularMomentum(const float *mass_d, const float3 *pos_d, const float3 *vel_d, float3 *totalAM_d, unsigned int particleCount) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int laneId = threadIdx.x & 31;
    unsigned int warpId = threadIdx.x >> 5;
    
    // Shared memory for warp partial sums (one float3 per warp)
    extern __shared__ float3 sharedAM[];
    
    // Each thread computes angular momentum for one particle
    float3 localAM = make_float3(0.0f, 0.0f, 0.0f);
    
    if (tid < particleCount) {
        float m = mass_d[tid];
        float3 r = pos_d[tid];
        float3 v = vel_d[tid];
        
        // Compute cross product r × v
        // L = m * (r × v)
        localAM.x = m * (r.y * v.z - r.z * v.y);
        localAM.y = m * (r.z * v.x - r.x * v.z);
        localAM.z = m * (r.x * v.y - r.y * v.x);
    }
    
    // Warp-level reduction using shuffle operations
    // Sum within each warp
    float3 warpSum = localAM;
    
    // Reduce within warp using shuffle operations
    warpSum.x += __shfl_down_sync(0xffffffff, warpSum.x, 16);
    warpSum.y += __shfl_down_sync(0xffffffff, warpSum.y, 16);
    warpSum.z += __shfl_down_sync(0xffffffff, warpSum.z, 16);
    
    warpSum.x += __shfl_down_sync(0xffffffff, warpSum.x, 8);
    warpSum.y += __shfl_down_sync(0xffffffff, warpSum.y, 8);
    warpSum.z += __shfl_down_sync(0xffffffff, warpSum.z, 8);
    
    warpSum.x += __shfl_down_sync(0xffffffff, warpSum.x, 4);
    warpSum.y += __shfl_down_sync(0xffffffff, warpSum.y, 4);
    warpSum.z += __shfl_down_sync(0xffffffff, warpSum.z, 4);
    
    warpSum.x += __shfl_down_sync(0xffffffff, warpSum.x, 2);
    warpSum.y += __shfl_down_sync(0xffffffff, warpSum.y, 2);
    warpSum.z += __shfl_down_sync(0xffffffff, warpSum.z, 2);
    
    warpSum.x += __shfl_down_sync(0xffffffff, warpSum.x, 1);
    warpSum.y += __shfl_down_sync(0xffffffff, warpSum.y, 1);
    warpSum.z += __shfl_down_sync(0xffffffff, warpSum.z, 1);
    
    // First thread of each warp stores its warp's partial sum
    if (laneId == 0) {
        sharedAM[warpId] = warpSum;
    }
    __syncthreads();
    
    // Number of warps in the block
    unsigned int numWarps = (blockDim.x + 31) >> 5;
    
    // Second phase: reduce the warp sums using thread 0
    if (threadIdx.x == 0) {
        float3 total = make_float3(0.0f, 0.0f, 0.0f);
        
        // Reduce all warp sums
        for (unsigned int i = 0; i < numWarps; i++) {
            total.x += sharedAM[i].x;
            total.y += sharedAM[i].y;
            total.z += sharedAM[i].z;
        }
        
        // Write final result to global memory
        totalAM_d->x = total.x;
        totalAM_d->y = total.y;
        totalAM_d->z = total.z;
    }
}