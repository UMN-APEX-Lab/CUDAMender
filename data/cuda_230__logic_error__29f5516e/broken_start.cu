#include "count_occurrences.h"

__global__ void k_countOccurrences(int * array1_d,
                                   int * array2_d,
                                   int len1,
                                   int len2,
                                   int * count_d) {
    extern __shared__ int shared_array2[];
    
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int globalIdx = blockIdx.x * blockSize + tid;
    int totalThreads = gridDim.x * blockSize;
    
    // Initialize count for elements this thread will process
    // First, we need to load array2 into shared memory in tiles
    
    int count = 0;
    int element1 = 0;
    bool hasElement = false;
    
    // Each thread may process multiple elements from array1_d
    for (int idx1 = globalIdx; idx1 < len1; idx1 += totalThreads) {
        element1 = array1_d[idx1];
        count = 0;
        
        // Process array2 in tiles using shared memory
        for (int tileStart = 0; tileStart < len2; tileStart += blockSize) {
            // Load tile of array2 into shared memory
            int loadIdx = tileStart + tid;
            if (loadIdx < len2) {
                shared_array2[tid] = array2_d[loadIdx];
            } else {
                shared_array2[tid] = element1 + 1; // Value that won't match
            }
            __syncthreads();
            
            // Compare element1 against all elements in the shared memory tile
            // Each thread checks all elements in the shared tile
            int tileSize = min(blockSize, len2 - tileStart);
            for (int j = 0; j < tileSize; j++) {
                if (shared_array2[j] == element1) {
                    count++;
                }
            }
            __syncthreads();
        }
        
        // Write result
        count_d[idx1] = count;
    }
}