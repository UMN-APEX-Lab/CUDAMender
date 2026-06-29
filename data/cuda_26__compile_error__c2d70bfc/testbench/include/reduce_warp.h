#ifndef REDUCE_WARP_H
#define REDUCE_WARP_H

#include <cuda.h>
#include "cuda_runtime.h"

__device__ float reduce_warp(float v);
__device__ float reduce_threadblock(float v, float *smem);

#endif