#include "include/bfs_kernel.h"

__global__ void k_bfsKernel(
    const int* rowOffsets_d,
    const int* colIndices_d,
    const int* frontier_d,
    int frontierSize,
    int currentLevel,
    int* distances_d,
    int* nextFrontier_d,
    int* nextFrontierSize_d)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= frontierSize) {
        return;
    }
    
    int node = frontier_d[tid];
    
    int start = rowOffsets_d[node];
    int end = rowOffsets_d[node + 1];
    
    for (int i = start; i < end; i++) {
        int neighbor = colIndices_d[i];
        
        // Check if neighbor is unvisited (distance is -1)
        // Using atomicCAS to avoid race conditions
        int oldDist = atomicCAS(&distances_d[neighbor], -1, currentLevel + 1);
        
        if (oldDist == -1) {
            // Neighbor was unvisited, add to next frontier
            int pos = atomicAdd(nextFrontierSize_d, 1);
            nextFrontier_d[pos] = neighbor;
        }
    }
}