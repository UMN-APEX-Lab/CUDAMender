#ifndef K_NEAREST_NEIGHBORS_H
#define K_NEAREST_NEIGHBORS_H

#include <cuda_runtime.h>

#define N_DIMS 3

__global__ void k_nearestNeighbors(const float* inputVectorA_d, int nA, float* inputVectorB_d, int nB, int* nearestNeighborIndex);

#endif // K_NEAREST_NEIGHBORS_H