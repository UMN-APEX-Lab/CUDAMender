#include "prefix_sum.h"

namespace cg = cooperative_groups;

__global__ void k_prefixSum(int* inputArray_d, int* outputArray_d, 
                           int* blockSums_d, int totalElementCount, int warpsPerBlock) {

    const int WARP_SIZE = warpSize;  // Built-in CUDA variable
    const int WARP_SHIFT = __ffs(WARP_SIZE) - 1;  // Calculate log2(warpSize)
    
    cg::grid_group grid = cg::this_grid();

    // Thread and block identification
    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int blockSize = blockDim.x;
    int globalThreadId = blockId * blockSize + threadId;
    int gridSize = gridDim.x * blockSize;

    // Warp-level identification
    int laneId = threadId & (WARP_SIZE - 1);    // Position within warp (0-31)
    int warpId = threadId >> WARP_SHIFT;        

    // Dynamic shared memory 
    extern __shared__ int dynamicSharedMem[];
    int* warpScanResults = dynamicSharedMem;
    int* cumulativeBlockOffset = &dynamicSharedMem[warpsPerBlock];

    // Initialize cumulative offset for this block
    if (threadId == 0) {
        *cumulativeBlockOffset = 0;
    }
    __syncthreads();

    for (int chunkStartIndex = 0; chunkStartIndex < totalElementCount; chunkStartIndex += gridSize) {
        int globalElementIndex = chunkStartIndex + globalThreadId;
        __syncthreads();

        // Load input value (zero-pad out-of-bounds elements)
        int currentValue = (globalElementIndex < totalElementCount) ? 
                          inputArray_d[globalElementIndex] : 0;

        // Perform warp-level inclusive scan using shuffle operations
        for (int scanOffset = 1; scanOffset < WARP_SIZE; scanOffset <<= 1) {
            int shuffledValue = __shfl_up_sync(0xffffffff, currentValue, scanOffset);
            if (laneId >= scanOffset) {
                currentValue += shuffledValue;
            }
        }

        // Store warp totals in shared memory
        if (laneId == (WARP_SIZE - 1)) {
            warpScanResults[warpId] = currentValue;
        }
        __syncthreads();

        // Warp 0 performs scan on warp totals
        if (warpId == 0) {
            int warpPrefix = (laneId < warpsPerBlock) ? warpScanResults[laneId] : 0;
            
            // Scan the warp totals
            for (int scanOffset = 1; scanOffset < WARP_SIZE; scanOffset <<= 1) {
                int shuffledPrefix = __shfl_up_sync(0xffffffff, warpPrefix, scanOffset);
                if (laneId >= scanOffset) {
                    warpPrefix += shuffledPrefix;
                }
            }
            
            if (laneId < warpsPerBlock) {
                warpScanResults[laneId] = warpPrefix;
            }
        }
        __syncthreads();

        // Add preceding warp contributions to current thread's value
        if (warpId > 0) {
            currentValue += warpScanResults[warpId - 1];
        }

        // Store block total for inter-block coordination
        if (threadId == (blockSize - 1)) {
            blockSums_d[blockId] = currentValue;
        }

        // Grid-wide synchronization - wait for all blocks
        grid.sync();

        // Block 0 performs exclusive scan on block totals
        if (blockId == 0 && threadId == 0) {
            int runningBlockSum = 0;
            for (int blockIndex = 0; blockIndex < gridDim.x; blockIndex++) {
                int currentBlockSum = blockSums_d[blockIndex];
                blockSums_d[blockIndex] = runningBlockSum;  // Exclusive prefix
                runningBlockSum += currentBlockSum;
            }
            blockSums_d[gridDim.x] = runningBlockSum;  // Total chunk sum
        }

        // Another grid-wide barrier
        grid.sync();

        // Write final inclusive scan result
        if (globalElementIndex < totalElementCount) {
            outputArray_d[globalElementIndex] = currentValue
                                              + blockSums_d[blockId]
                                              + *cumulativeBlockOffset;
        }

        // Update cumulative offset for next chunk
        __syncthreads();
        if (threadId == 0) {
            *cumulativeBlockOffset += blockSums_d[gridDim.x];
        }
        __syncthreads();

        // Final grid synchronization before processing next chunk
        grid.sync();
    }
}