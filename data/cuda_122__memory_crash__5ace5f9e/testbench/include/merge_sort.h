#ifndef MERGE_SORT_H
#define MERGE_SORT_H

// This device function divides the input array into n blocks 
// and sorts the elements within each block using merge sort method.
// This kernel uses shared memory to store the temporary sorted blocks.
__device__ void d_mergeSortWithinBlock(float *input_d, float *sortedBlocks_d, int numElements);

// This device function merges the sorted blocks into a single sorted array.
// Requires grid-wide synchronization to coordinate across all blocks.
// Must use CUDA cooperative groups for grid-level synchronization.
__device__ void d_mergeSortAcrossBlocks(float *sortedBlocks_d, float *output_d, int numElements);

// This kernel will sort the elements using merge sort technique by calling two device functions:
// d_mergeSortWithinBlock and d_mergeSortAcrossBlocks
// This kernel must be launched with cudaLaunchCooperativeKernel to enable grid-wide synchronization.
__global__ void k_mergeSort(float *input_d, float *sortedBlocks_d, float *output_d, int numElements);

#endif // MERGE_SORT_H