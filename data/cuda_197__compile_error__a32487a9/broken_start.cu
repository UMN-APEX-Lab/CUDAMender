#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void k_compactElementsOfSegmentsWithThreshold(
    int numSegments,
    float* array_d,
    float threshold,
    float defaultValue
) {
    int segmentId = blockIdx.x;
    int laneId = threadIdx.x;
    
    if (segmentId >= numSegments) return;
    
    // Each segment is 32 elements, one warp per segment
    // Load the element for this lane
    float val = array_d[segmentId * 32 + laneId];
    
    // Compute predicate: element > threshold
    bool pred = (val > threshold);
    
    // Compute ballot to get bitmask of active lanes
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, pred);
    
    // Count bits set before this lane's position to get output position
    unsigned int mask = (1U << laneId) - 1;
    unsigned int prefix = __popc(ballot & mask);
    
    // Total count of elements that pass the predicate
    unsigned int total = __popc(ballot);
    
    // Determine output value and position
    float output;
    int outPos;
    
    if (pred) {
        // This lane has a valid element, place it at prefix position
        output = val;
        outPos = prefix;
    } else {
        // This lane doesn't have valid element, will be filled with default
        // Position is after all valid elements
        output = defaultValue;
        outPos = total + (laneId - prefix); // offset into default section
    }
    
    // Write to output position in segment
    // Use shuffle to handle the case where we need to write from different lanes
    // Actually, each thread writes to its computed position directly
    
    // We need to handle the case where multiple threads write to same location
    // Instead, use warp shuffle to redistribute values
    
    // First, let each thread know where its value should go
    // Then use shuffle to get the value that belongs at this thread's position
    
    // Alternative: use shared memory or direct shuffle-based scatter
    
    // Let's use a different approach: each thread computes which source lane
    // should provide the value for its destination position
    
    // For position laneId, find which source lane has data for it
    // If laneId < total, find the lane with prefix == laneId and pred == true
    // Else, find a lane with pred == false in order
    
    // Use binary search on ballot to find source lane
    int srcLane = -1;
    
    if (laneId < (int)total) {
        // Need to find lane where prefix == laneId and pred == true
        // This is the (laneId+1)-th set bit in ballot
        srcLane = __fns(ballot, 0, laneId + 1);
    } else {
        // Need to find lane among those with pred == false
        // Position within false elements: laneId - total
        unsigned int falseBallot = ~ballot & 0xFFFFFFFF;
        int falsePos = laneId - total;
        srcLane = __fns(falseBallot, 0, falsePos + 1);
    }
    
    // Shuffle the value from source lane
    float result = __shfl_sync(0xFFFFFFFF, val, srcLane);
    
    // If source lane had pred==false, use defaultValue
    bool srcPred = ((ballot >> srcLane) & 1U) != 0;
    if (!srcPred) {
        result = defaultValue;
    }
    
    // Write result
    array_d[segmentId * 32 + laneId] = result;
}