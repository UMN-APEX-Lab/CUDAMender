// file: solution.cu
#include "k_sortSegments.h"
#include <cfloat>
#include <cmath>

__global__ void k_sortSegments(float *array_d, float *arrayOut_d, int segmentSize, int arraySize) {
    unsigned int activeWarpLanesMask = __activemask();
    unsigned int numActiveWarpLanes = __popc(activeWarpLanesMask);
    constexpr float ERROR_TOLERANCE = 1e-9f;
    int blockSize = blockDim.x;
    int localThreadIndex = threadIdx.x;
    int warpLane = localThreadIndex % warpSize;
    int numWarpsPerBlock = (blockSize + warpSize - 1) / warpSize;
    int warpIndex = blockIdx.x * numWarpsPerBlock + localThreadIndex / warpSize;
    int numTotalWarpsActive = numWarpsPerBlock * gridDim.x;
    int numWarpsRequired = arraySize/segmentSize;
    int warpStrideIterations = (segmentSize + numActiveWarpLanes - 1) / numActiveWarpLanes;
    int gridStrideIterations = (numWarpsRequired + numTotalWarpsActive - 1) / numTotalWarpsActive;
    // Grid stride loops enable flexibility with computable array sizes.
    for(int gridIteration = 0; gridIteration < gridStrideIterations; gridIteration++) {
        int gridOffset = segmentSize * (gridIteration * numTotalWarpsActive + warpIndex);
        // Iterating the data for the left side of comparisons.
        for(int warpIteration1 = 0; warpIteration1 < warpStrideIterations; warpIteration1++) {
            int data1Index = warpIteration1 * numActiveWarpLanes + warpLane;
            float data1 = FLT_MAX;
            bool isData1Valid = (data1Index < segmentSize) && (gridOffset + data1Index < arraySize);
            int rank = 0;
            if(isData1Valid) {
                data1 = array_d[gridOffset + data1Index];
            }
            // Iterating the data for the right side of comparisons.
            for(int warpIteration2 = 0; warpIteration2 < warpStrideIterations; warpIteration2++) {
                int data2Index = warpIteration2 * numActiveWarpLanes + warpLane;
                int numElementsToCompare = segmentSize - warpIteration2 * numActiveWarpLanes;
                numElementsToCompare = (numElementsToCompare < numActiveWarpLanes ? numElementsToCompare : numActiveWarpLanes);
                float data2 = FLT_MAX;
                if(data2Index < segmentSize && gridOffset + data2Index < arraySize) {
                    data2 = array_d[gridOffset + data2Index];
                }
                // Iterating all comparisons using warp-shuffle.
                for(int i = 0; i < numElementsToCompare; i++) {
                    float comparedData = __shfl_sync(activeWarpLanesMask, data2, i);
                    bool case1 = comparedData < data1;
                    bool case2 = fabsf(comparedData - data1) < ERROR_TOLERANCE;
                    bool case3 = data1Index > (warpIteration2 * numActiveWarpLanes + i);
                    rank += ((case1 || (case2 && case3)) ? 1 : 0);
                }
            }
            if(isData1Valid && gridOffset + rank < arraySize) {
                arrayOut_d[gridOffset + rank] = data1;
            }
        }
    }
}
