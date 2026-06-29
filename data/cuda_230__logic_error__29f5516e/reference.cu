#include "count_occurrences.h"

__global__ void k_countOccurrences(int * array1_d, 
                                   int * array2_d, 
                                   int len1, 
                                   int len2, 
                                   int * count_d) {
    int localThreadIndex = threadIdx.x;
    int numLocalThreads = blockDim.x;
    int globalThreadIndex = localThreadIndex + blockIdx.x * numLocalThreads;
    int numGlobalThreads = numLocalThreads * gridDim.x;
    int numActiveBlocks = (len1 + numLocalThreads - 1) / numLocalThreads;
    extern __shared__ int s_array2[];
    // Grid-stride loop for integers to count.
    int numArray1Iterations = (len1 + numGlobalThreads - 1) / numGlobalThreads;
    for (int array1Iteration = 0; array1Iteration < numArray1Iterations; array1Iteration++) {
        int array1Index = array1Iteration * numGlobalThreads + globalThreadIndex;
        int array1Element = (array1Index < len1 ? array1_d[array1Index] : 0);
        int numArray2Iterations = (len2 + numLocalThreads - 1) / numLocalThreads;
        if(array1Index / numLocalThreads < numActiveBlocks) {
            int count = 0;
            for (int array2Iteration = 0; array2Iteration < numArray2Iterations; array2Iteration++) {
                int array2Offset = array2Iteration * numLocalThreads;
                int array2Index = array2Offset + localThreadIndex;
                s_array2[localThreadIndex] = (array2Index < len2 ? array2_d[array2Index] : 0);
                __syncthreads();
                for (int i = 0; i < numLocalThreads; i++) {
                    if (i + array2Offset < len2) {
                        count += (array1Element == s_array2[i]);
                    } else {
                        break;
                    }
                }
                __syncthreads();
            }
            if (array1Index < len1) {
                count_d[array1Index] = count;
            }
        }
    }
}