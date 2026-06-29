#include <cuda_runtime.h>

#define NUM_DEVICE_MEMORY_ELEM (1024)

__device__ float plateCurrentTemperatures_d[NUM_DEVICE_MEMORY_ELEM];
__device__ float alternateBuffer_d[NUM_DEVICE_MEMORY_ELEM];