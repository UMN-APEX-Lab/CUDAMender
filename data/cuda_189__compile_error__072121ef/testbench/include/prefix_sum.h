#ifndef PREFIX_SUM_H
#define PREFIX_SUM_H

#include <cuda_runtime.h>
#include <cooperative_groups.h>

__global__ void k_prefixSum(int* inputArray_d, int* outputArray_d, 
                           int* blockSums_d, int totalElementCount, int warpsPerBlock);

#endif // PREFIX_SUM_H