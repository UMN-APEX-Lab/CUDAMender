// file: solution.cu
#include "k_nearest_neighbors.h"
#include <float.h>

__global__ void k_nearestNeighbors(const float* inputVectorA_d, int nA, float* inputVectorB_d, int nB, int* nearestNeighborIndex) {
    // Since the inputVectorB will be repeatedly accessed by inputVectorA, copying the inputVectorB into shared memory will reduce memory latency
    // Allocating dynamic sized shared memory
    extern __shared__ float sharedVectorB[];

    // Threads copy inputVectorB into shared memory. 
    size_t sharedIdx = threadIdx.x;
    while (sharedIdx < N_DIMS * nB) {
        sharedVectorB[sharedIdx] = inputVectorB_d[sharedIdx];
        sharedIdx += blockDim.x;
    }
    __syncthreads();

    // Distance computation from a point in inputVectorA to every point in inputVectorB
    for (int tIdx = blockIdx.x * blockDim.x + threadIdx.x; tIdx < nB; tIdx += gridDim.x * blockDim.x) {
        // Input Vector
        float ptA[N_DIMS];
        for (int nd = 0; nd < N_DIMS; nd++) {
            ptA[nd] = inputVectorA_d[N_DIMS * tIdx + nd];
        }
        
        // Setting distance value to MAX and index to zero
        float minDist = FLT_MAX;
        size_t minDistIdx = 0;
        for (int i = 0; i < nB; i++) {
            // Computing distance from inputVectorA to all N-D points in inputVectorB
            float sqDist = 0;
            for (int nd = 0; nd < N_DIMS; nd++) {
                float dist = ptA[nd] - sharedVectorB[N_DIMS * i + nd];
                sqDist += (dist * dist);
            }
            
            // Finding minimum distance and capturing their indices.
            if ((fabs(sqDist - minDist) > FLT_EPSILON) && (sqDist < minDist)) {
                minDist = sqDist;
                minDistIdx = i;
            }
        }

        // Storing the outputs
        nearestNeighborIndex[tIdx] = minDistIdx;
    }
}
