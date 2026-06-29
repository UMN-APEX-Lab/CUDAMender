#ifndef ANGULAR_MOMENTUM_H
#define ANGULAR_MOMENTUM_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                                   \
do {                                                                                       \
    cudaError_t error = call;                                                              \
    if(error != cudaSuccess) {                                                             \
        fprintf(stderr,                                                                    \
            "CUDA Error: %s at %s:%d\n",                                                   \
            cudaGetErrorString(error),                                                     \
            __FILE__,                                                                      \
            __LINE__);                                                                     \
        exit(error);                                                                       \
    }                                                                                      \
} while(0)

// CUDA kernel to compute total angular momentum using warp-level reduction
__global__ void k_computeAngularMomentum(const float *mass_d, const float3 *pos_d, const float3 *vel_d, float3 *totalAM_d, unsigned int particleCount);

#endif // ANGULAR_MOMENTUM_H