#include <cooperative_groups.h>
#include "temperature_distribution.h"

using namespace cooperative_groups;

__global__ void k_temperatureDistribution(float *temperatureValues, int numPlateElementsX, int numPlateElementsY, int numIterations) {
    extern __shared__ float sharedMem[];
    
    // Get thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int totalElements = numPlateElementsX * numPlateElementsY;
    int idx = y * numPlateElementsX + x;
    
    // Create grid-wide group for synchronization
    grid_group g = this_grid();
    
    // Check if thread is within bounds
    if (idx >= totalElements) return;
    
    // Check if this is a boundary element (should not be updated)
    bool isBoundary = (x == 0 || x == numPlateElementsX - 1 || 
                       y == 0 || y == numPlateElementsY - 1);
    
    // Get pointers to the two buffers (stored in shared memory for efficiency)
    float *currentBuffer = temperatureValues;
    float *alternateBuffer = &temperatureValues[totalElements];
    
    // For each iteration
    for (int iter = 0; iter < numIterations; iter++) {
        // Synchronize all threads before starting iteration
        g.sync();
        
        if (!isBoundary) {
            // Calculate neighbor indices
            int top = (y - 1) * numPlateElementsX + x;
            int bottom = (y + 1) * numPlateElementsX + x;
            int left = y * numPlateElementsX + (x - 1);
            int right = y * numPlateElementsX + (x + 1);
            
            // Calculate new temperature as average of neighbors
            float newTemp = (currentBuffer[top] + currentBuffer[bottom] +
                            currentBuffer[left] + currentBuffer[right]) / 4.0f;
            
            // Write to alternate buffer
            alternateBuffer[idx] = newTemp;
        } else {
            // Boundary values remain the same, copy to alternate buffer
            alternateBuffer[idx] = currentBuffer[idx];
        }
        
        // Synchronize all threads after writing
        g.sync();
        
        // Swap buffers for next iteration
        float *temp = currentBuffer;
        currentBuffer = alternateBuffer;
        alternateBuffer = temp;
    }
    
    // Final sync before writing results back
    g.sync();
    
    // Copy results from current buffer back to original array
    if (idx < totalElements) {
        temperatureValues[idx] = currentBuffer[idx];
    }
}