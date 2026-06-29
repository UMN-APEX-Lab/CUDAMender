#include "kernel_contract.h"

__global__ void k_compactElementsOfSegmentsWithThreshold(int numSegments, float* array_d, float threshold, float defaultValue) {
    constexpr unsigned int ALL_LANES_REQUIRE_COMPACTION = 0xFFFFFFFF;
    constexpr unsigned int ALL_LANES_VOTE = 0xFFFFFFFF;
    int numTotalThreads = blockDim.x * gridDim.x;
    int numTotalWarps = numTotalThreads / warpSize;
    int localThreadIndex = threadIdx.x;
    int threadIndex = localThreadIndex + blockIdx.x * blockDim.x;
    int localWarpIndex = localThreadIndex / warpSize;
    int localSegmentOffset = localWarpIndex * SEGMENT_SIZE;
    int warpIndex = threadIndex / warpSize;
    int warpLane = threadIdx.x % warpSize;
    int numStepsForGridStride = (numSegments + numTotalWarps - 1) / numTotalWarps;
    int arrayBorder = numSegments * SEGMENT_SIZE;
    unsigned int currentWarpLaneMask = 1 << warpLane;
    // Calculating the mask to find the number of elements before the current warp lane that have an output.
    unsigned int maskForCount = currentWarpLaneMask - 1;
    extern __shared__ float s_output[];
    for (int i = 0; i < numStepsForGridStride; i++) {
        // Clearing the output buffer.
        s_output[localThreadIndex] = defaultValue;
        __syncwarp();
        int segmentIndex = i * numTotalWarps + warpIndex;
        int segmentOffset = SEGMENT_SIZE * segmentIndex;
        int globalElementIndex = segmentOffset + warpLane;
        float data = globalElementIndex < arrayBorder ? array_d[globalElementIndex] : defaultValue;
        int predicate = data > threshold;
        // Identifying which warp lanes have an output.
        unsigned int predicateMask = __ballot_sync(ALL_LANES_VOTE, predicate);
        if (predicateMask != ALL_LANES_REQUIRE_COMPACTION) {
            // Calculating the number of elements to compact before the current warp lane. This number is used as the offset within the current segment output.
            unsigned int offset = __popc(predicateMask & maskForCount);
            // Scattering the results to the shared memory output using the offset.
            if (predicate) {
                s_output[localSegmentOffset + offset] = data;
            }
            __syncwarp();
            // Uniformly writing the output from shared memory to global memory.
            if (globalElementIndex < arrayBorder) {
                array_d[globalElementIndex] = s_output[localThreadIndex];
            }
        }
    }
}