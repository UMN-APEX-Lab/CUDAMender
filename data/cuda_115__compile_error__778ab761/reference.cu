#include "odd_even_separation.h"

__global__ void k_separateOddEven(int *input_d, int *oddData_d, int *evenData_d, int numElements) {
    __shared__ int s_input[NUM_THREADS_PER_BLOCK];
    int localThreadId = threadIdx.x;
    int threadId = localThreadId + blockIdx.x * blockDim.x;
    int strideSize = blockDim.x * gridDim.x;
    int numStrideLoopIterations = 1 + (numElements - 1) / strideSize;
    int halfNumberOfBlockThreads = NUM_THREADS_PER_BLOCK >> 1;
    int numElementsDiv2 = numElements >> 1;
    int numDestinationElements = (numElements & 1) ? (numElementsDiv2 + 1) : numElementsDiv2;
    int sourceId;
    int destinationOffset;

    // If the current warp is in the first half of block, then working on the even-index output without divergence.
    // If the current warp is in the second half of block, then working on odd-index output without divergence.
    if(localThreadId < halfNumberOfBlockThreads) {
        sourceId = (localThreadId << 1);
        destinationOffset = localThreadId;
    } else {
        sourceId = ((localThreadId - halfNumberOfBlockThreads) << 1) + 1;
        destinationOffset = localThreadId - halfNumberOfBlockThreads;
    }

    for(int stride = 0; stride < numStrideLoopIterations; stride++) {
        int itemId = threadId + stride * strideSize;
        int itemBlockStart = itemId - (itemId % NUM_THREADS_PER_BLOCK);
        int destinationId = (itemBlockStart >> 1) + destinationOffset;
        
        // Loading data in a coalesced & uniform manner.
        if(itemId < numElements) {
            s_input[localThreadId] = input_d[itemId];
        }
        
        __syncthreads();
        
        // Storing data in a coalesced manner while doing interleaved access on the shared memory instead of the global memory of the input.
        if (destinationId < numDestinationElements && sourceId < NUM_THREADS_PER_BLOCK) {
            // All warps in the first half of the block are writing to the evenData_d while other warps are writing to the oddData_d arrays, to avoid warp-divergence.
            if(localThreadId < halfNumberOfBlockThreads){
                evenData_d[destinationId] = s_input[sourceId];
            } else {
                oddData_d[destinationId] = s_input[sourceId];
            }
        }
        
        __syncthreads();
    }
}