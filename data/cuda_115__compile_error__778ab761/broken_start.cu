#include "odd_even_separation.h"

__global__ void k_separateOddEven(int *input_d, int *oddData_d, int *evenData_d, int numElements) {
    __shared__ int sharedData[NUM_THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Load data into shared memory with coalesced access
    if (gid < numElements) {
        sharedData[tid] = input_d[gid];
    }
    __syncthreads();
    
    // Calculate how many elements this block processes
    int blockStart = blockIdx.x * blockDim.x;
    int blockEnd = min(blockStart + blockDim.x, numElements);
    int elementsInBlock = blockEnd - blockStart;
    
    // Count odd and even positions in this block
    int numOddInBlock = elementsInBlock / 2;
    int numEvenInBlock = (elementsInBlock + 1) / 2;
    
    // Calculate output offsets for this block
    // Even indices come from positions 0, 2, 4, ... -> go to evenData_d
    // Odd indices come from positions 1, 3, 5, ... -> go to oddData_d
    
    // Global offset calculation: number of even/odd elements before this block
    int evenOffsetBefore = (blockStart + 1) / 2;  // ceil(blockStart / 2) for even positions
    int oddOffsetBefore = blockStart / 2;          // floor(blockStart / 2) for odd positions
    
    // Each thread processes one element from shared memory
    if (tid < elementsInBlock) {
        int value = sharedData[tid];
        int globalPos = blockStart + tid;
        
        // Avoid warp divergence by computing both cases
        bool isOddPos = (globalPos & 1);  // true if odd index (1, 3, 5, ...)
        
        // For even positions (0, 2, 4, ...): output to evenData_d
        // For odd positions (1, 3, 5, ...): output to oddData_d
        
        // Compute local index within this block's even/odd elements
        int localEvenIdx = tid / 2;           // 0,0,1,1,2,2,...
        int localOddIdx = (tid - 1) / 2;      // -1,0,0,1,1,2,... (invalid for tid=0)
        
        // Write to appropriate output array without divergence
        if (isOddPos) {
            // Odd position: write to oddData_d
            oddData_d[oddOffsetBefore + localOddIdx] = value;
        } else {
            // Even position: write to evenData_d
            evenData_d[evenOffsetBefore + localEvenIdx] = value;
        }
    }
}