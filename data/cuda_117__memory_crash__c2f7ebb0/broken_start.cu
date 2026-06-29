#include "k_nearest_neighbors.h"
#include <float.h>

__global__ void k_nearestNeighbors(const float* inputVectorA_d, int nA, float* inputVectorB_d, int nB, int* nearestNeighborIndex) {
    // Each thread handles one point from array A
    int idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= nA) return;

    // Load the current point from A into registers
    float ax = inputVectorA_d[idxA * N_DIMS + 0];
    float ay = inputVectorA_d[idxA * N_DIMS + 1];
    float az = inputVectorA_d[idxA * N_DIMS + 2];

    float minDistanceSq = FLT_MAX;
    int bestIndex = -1;

    // Use shared memory to cache chunks of array B to reduce global memory pressure
    // We process B in tiles. Each tile consists of blockDim.x points from B.
    extern __shared__ float s_B[]; // Size should be blockDim.x * N_DIMS

    for (int tileStart = 0; tileStart < nB; tileStart += blockDim.x) {
        // Collaborative load of array B into shared memory
        int loadIdx = threadIdx.x;
        if (tileStart + loadIdx < nB) {
            int bOffset = (tileStart + loadIdx) * N_DIMS;
            s_B[loadIdx * N_DIMS + 0] = inputVectorB_d[bOffset + 0];
            s_B[loadIdx * N_DIMS + 1] = inputVectorB_d[bOffset + 1];
            s_B[loadIdx * N_DIMS + 2] = inputVectorB_d[bOffset + 2];
        }
        __syncthreads();

        // Compare point A with all points currently in shared memory
        int currentTileSize = (nB - tileStart < blockDim.x) ? (nB - tileStart) : blockDim.x;
        for (int i = 0; i < currentTileSize; ++i) {
            float bx = s_B[i * N_DIMS + 0];
            float by = s_B[i * N_DIMS + 1];
            float bz = s_B[i * N_DIMS + 2];

            float dx = ax - bx;
            float dy = ay - by;
            float dz = az - bz;
            float distSq = dx * dx + dy * dy + dz * dz;

            if (distSq < minDistanceSq) {
                minDistanceSq = distSq;
                bestIndex = tileStart + i;
            }
        }
        __syncthreads();
    }

    nearestNeighborIndex[idxA] = bestIndex;
}

// Note: The problem description requests a kernel implementation. 
// Since the kernel uses dynamic shared memory based on blockDim.x, 
// the caller must launch it with: 
// k_nearestNeighbors<<<grid, block, block.x * N_DIMS * sizeof(float)>>>(...);
// However, based on the provided signature and requirements, the logic above 
// implements the shared memory tiling pattern requested.
