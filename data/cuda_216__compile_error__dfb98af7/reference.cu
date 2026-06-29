#include "bfs_kernel.h"

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
    // Calculate the global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Use a grid-stride loop to allow each thread to process multiple nodes
    // if the frontier is larger than the number of threads in the grid.
    for (int i = tid; i < frontierSize; i += gridDim.x * blockDim.x) {
        // Get the current node 'u' from the frontier to process
        int u = frontier_d[i];

        // Get the range of neighbors for node 'u' from the CSR representation
        int start = rowOffsets_d[u];
        int end = rowOffsets_d[u + 1];

        // Iterate over the neighbors of 'u'
        for (int j = start; j < end; ++j) {
            int v = colIndices_d[j];

            // Check if the neighbor 'v' has not been visited yet (distance is -1)
            if (distances_d[v] == -1) {
                // Atomically attempt to set the distance of 'v'.
                // atomicCAS (Compare-And-Swap) checks if distances_d[v] is still -1.
                // If it is, it sets the distance to `currentLevel + 1` and returns the old value (-1).
                // This ensures that only the first thread to reach 'v' in this level will process it.
                if (atomicCAS(&distances_d[v], -1, currentLevel + 1) == -1) {
                    // Atomically get a unique index in the next frontier array.
                    // atomicAdd increments `nextFrontierSize_d` and returns its value before the increment.
                    int pos = atomicAdd(nextFrontierSize_d, 1);

                    // Add the newly discovered node 'v' to the next frontier.
                    nextFrontier_d[pos] = v;
                }
            }
        }
    }
}