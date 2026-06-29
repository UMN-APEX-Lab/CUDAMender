#include "angular_momentum.h"

__global__ void k_computeAngularMomentum(const float *mass_d, const float3 *pos_d, const float3 *vel_d, float3 *totalAM_d, unsigned int particleCount) {
    unsigned int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride   = blockDim.x * gridDim.x;

    float3 L = make_float3(0.0f, 0.0f, 0.0f);

    for (unsigned int i = threadId; i < particleCount; i += stride) {
        float m = mass_d[i];
        float3 pos = pos_d[i];
        float3 vel = vel_d[i];

        float3 tmp;
        tmp.x = m * (pos.y * vel.z - pos.z * vel.y);
        tmp.y = m * (pos.z * vel.x - pos.x * vel.z);
        tmp.z = m * (pos.x * vel.y - pos.y * vel.x);

        L.x += tmp.x;
        L.y += tmp.y;
        L.z += tmp.z;
    }

    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        L.x += __shfl_down_sync(mask, L.x, offset);
        L.y += __shfl_down_sync(mask, L.y, offset);
        L.z += __shfl_down_sync(mask, L.z, offset);
    }

    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(&(totalAM_d->x), L.x);
        atomicAdd(&(totalAM_d->y), L.y);
        atomicAdd(&(totalAM_d->z), L.z);
    }
}