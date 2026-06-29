#ifndef TEMPERATURE_DISTRIBUTION_H
#define TEMPERATURE_DISTRIBUTION_H

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define NUM_DEVICE_MEMORY_ELEM (1024)

__global__ void k_temperatureDistribution(float *temperatureValues, int numPlateElementsX, int numPlateElementsY, int numIterations);

extern __device__ float plateCurrentTemperatures_d[NUM_DEVICE_MEMORY_ELEM];
extern __device__ float alternateBuffer_d[NUM_DEVICE_MEMORY_ELEM];

#endif // TEMPERATURE_DISTRIBUTION_H