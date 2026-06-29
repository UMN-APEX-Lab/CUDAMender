#include "temperature_distribution.h"

namespace cg = cooperative_groups;

__global__ void k_temperatureDistribution(float *temperatureValues, int numPlateElementsX, int numPlateElementsY, int numIterations) {
    // Cooperative Grid Group
    cg::grid_group grid = cg::this_grid();

    float *src = temperatureValues;
    float *dst = alternateBuffer_d;

    // Global column index
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    // Global row index
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if ((xIndex >= numPlateElementsX) || (yIndex >= numPlateElementsY)) {
        return;
    }      

    for (int iter = 0; iter < numIterations; iter++) {
        // Compute new temperature values
        if ((xIndex > 0) && (xIndex < numPlateElementsX - 1) && (yIndex > 0) && (yIndex < numPlateElementsY - 1)) {
            float topVal    = src[(yIndex - 1) * numPlateElementsX + xIndex];
            float bottomVal = src[(yIndex + 1) * numPlateElementsX + xIndex];
            float leftVal   = src[yIndex * numPlateElementsX + (xIndex - 1)];
            float rightVal  = src[yIndex * numPlateElementsX + (xIndex + 1)];

            float updatedTemperature = (topVal + bottomVal + leftVal + rightVal) / 4;
            dst[yIndex * numPlateElementsX + xIndex] = updatedTemperature;
        }

        // Ensure computation is complete across all blocks before loading values for next itertation
        grid.sync();
        
        // Swap the read and write buffers
        float *temp = src;
        src = dst;
        dst = temp;
    }
    
    if((numIterations % 2) == 1) {
        // Write updated values back to global memory
        temperatureValues[yIndex * numPlateElementsX + xIndex] = src[yIndex * numPlateElementsX + xIndex];
    }
}