// file: solution.cu
#include "merge_sort.h"
#include "cuda_common.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__device__ void d_mergeSortWithinBlock(float *input_d, float *sortedBlocks_d, int numElements) {
    // Each block handles a portion of the array
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x;

    // Create shared memory for the block to work on
    extern __shared__ float sharedData[];

    // If the thread id is within bounds, load data into shared memory
    if (threadid < numElements) {
        sharedData[threadIdx.x] = input_d[threadid];
    } else {
        sharedData[threadIdx.x] = MAXVALUE;  // Fill with maximum value if out of bounds
    }

    // Synchronize threads in the block
    __syncthreads();

    // Perform the merge sort using shared memory
    for (int divLen = 1; divLen <= blockSize / 2; divLen *= 2) {
        // Thread ID within the block
        int leftIdx = threadIdx.x * 2 * divLen + blockIdx.x * blockDim.x;
        int blockStride = blockSize + blockIdx.x * blockDim.x;
        int rightIdx = leftIdx + divLen;
        int endIdx = leftIdx + 2 * divLen;

        // Perform a merge of the two parts
        if (leftIdx < blockStride) {
            int i = leftIdx;
            int j = rightIdx;

            // Merge the elements
            for (int k = leftIdx; k < endIdx && k < numElements; k++) {
                float iValue = sharedData[i % blockSize];
                float jValue = sharedData[j % blockSize];

                // Merge in sorted order
                if (i < rightIdx && (j >= endIdx || iValue <= jValue)) {
                    sortedBlocks_d[k] = iValue;
                    i++;
                } else {
                    sortedBlocks_d[k] = jValue;
                    j++;
                }
            }
        }

        // Synchronize threads after each merge step
        __syncthreads();

        // Copy the sorted data from shared memory back to the output array
        if (threadid < numElements) {
            sharedData[threadIdx.x] = sortedBlocks_d[threadid];
        }

        __syncthreads();
    }
}

__device__ void d_mergeSortAcrossBlocks(float *sortedBlocks_d, float *output_d, int numElements) {
    auto grid = cooperative_groups::this_grid();
    
    //load the sortedBlocks in to the output
    for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numElements; idx += gridDim.x * blockDim.x){
        output_d[idx] = sortedBlocks_d[idx];
    }
    
    // Ensure all blocks finish writing to output_d before any block starts reading from it
    grid.sync();
    
    // iterate each sorted block index from sortedBlocks_d and merge in to mergeSortedBlocks 
    for(int blockStride = blockDim.x; blockStride <= numElements; blockStride *= 2){
        // calculate the left block start and end index.
        int lblockStartIdx = 2 * blockIdx.x * blockStride;
        int lblockEndIdx = min(numElements, blockStride + 2 * blockIdx.x * blockStride);
        // calculate the right block start and end index.  
        int rblockStartIdx = min(numElements, blockStride + 2 * blockIdx.x * blockStride);
        int rblockEndIdx = min(numElements, blockStride * 2 + 2 * blockIdx.x * blockStride);
        // initialize the left block and right block iterators 
        int i = lblockStartIdx, j = rblockStartIdx;
        for(int k = lblockStartIdx; k < rblockEndIdx; k++){
            // compare and merge the left blocks and right blocks data in to mergeSortedBlocks
            if( i < lblockEndIdx && j < rblockEndIdx){
                float lblockElement = output_d[i];
                float rblockElement = output_d[j];
                if(lblockElement < rblockElement){
                    sortedBlocks_d[k] = lblockElement;
                    i += 1;
                } else {
                    sortedBlocks_d[k] = rblockElement;
                    j += 1;
                }
            } else if ( i < lblockEndIdx) {
                float lblockElement = output_d[i]; 
                sortedBlocks_d[k] = lblockElement;
                i += 1;
            } else {
                float rblockElement = output_d[j];
                sortedBlocks_d[k] = rblockElement;
                j += 1;
            }
        }

        // synchronize all the threads in the grid after the merge of sorted block.
        grid.sync();
        
        // copy the mergeSortedBlocks in to output array.
        for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numElements; idx += gridDim.x * blockDim.x){
            output_d[idx] = sortedBlocks_d[idx];
        }
        // synchronize all the threads in the grid after the copy of sortedblocks to output.
        grid.sync();
    }
}

// This module implements the merge sort by calling two device functions
__global__ void k_mergeSort(float *input_d, float *sortedBlocks_d, float *output_d, int numElements){
    d_mergeSortWithinBlock(input_d, sortedBlocks_d, numElements);
    d_mergeSortAcrossBlocks(sortedBlocks_d, output_d, numElements);
}
