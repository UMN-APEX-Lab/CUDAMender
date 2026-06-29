#include "broadcast_tree.h"

__device__ __forceinline__ uint32_t d_broadcast(uint32_t messageId, 
                                                int warpLane, 
                                                uint32_t * message_d, 
                                                int numCommunicatingWarps, 
                                                int globalWarpId) {
    constexpr int BINARY_TREE_PATTERN_MULTIPLIER = 2;
    constexpr int BINARY_TREE_LEFT_CHILD_NODE = 1;
    constexpr int BINARY_TREE_RIGHT_CHILD_NODE = 2;
    int destinationBinaryTreeNodeOffset = globalWarpId * BINARY_TREE_PATTERN_MULTIPLIER;
    // Current warp is interpreted as the current parent node.
    int readIndex = globalWarpId * warpSize + warpLane;
    // The next two nodes in binary tree indexing are the child nodes of the parent.
    int leftChildNode = destinationBinaryTreeNodeOffset + BINARY_TREE_LEFT_CHILD_NODE;
    int rightChildNode = destinationBinaryTreeNodeOffset + BINARY_TREE_RIGHT_CHILD_NODE;
    int writeIndex1 = leftChildNode * warpSize + warpLane;
    int writeIndex2 = rightChildNode * warpSize + warpLane;
    while(true) {
        // Receiving data from a warp.
        uint32_t message = atomicExch(&message_d[readIndex], 0); 
        if((message & ID_MASK) == messageId) {
            // Sending data to two additional warps.
            if (leftChildNode < numCommunicatingWarps) {
                atomicExch(&message_d[writeIndex1], message);
            }
            if (rightChildNode < numCommunicatingWarps) {
                atomicExch(&message_d[writeIndex2], message);
            }
            // Decoding the data by shifting the id bits out.
            return message >> ID_BITS;
        }
    }
}

__global__ void k_broadcastWithHierarchicalPath(uint32_t input, 
                                                uint32_t * message_d, 
                                                int numCommunicatingWarps, 
                                                int * output_d) {
    int globalThreadId = threadIdx.x + blockIdx.x * blockDim.x;
    int globalWarpId = globalThreadId / warpSize;
    int warpLane = globalThreadId % warpSize;
    // A non-zero message index between 1 and 255 is required to work. Sample message id is chosen as a 1-based warp lane index. Any two warps in communication must have same message id in their corresponding warp lanes to receive/send data properly.
    uint32_t sampleMessageIndex = warpLane + 1;  
    bool broadcaster = globalWarpId == 0;
    bool warpIsInTheHierarchy = globalWarpId < numCommunicatingWarps;
    // Sending a message (thread index + input) from the first warp. The datas are 0, 1, 2, ..., 31.
    uint32_t sampleData = globalThreadId + input;
    uint32_t messageId = sampleMessageIndex;
    // Only the first warp requires preparation of the message as it is the broadcaster.
    if (broadcaster) {
        // Encoding messageId and sample data. Sample data is written to the most significant 24 bits of the message, and the message ID is written to the least significant 8 bits of the message. This is for atomically sending both id and data.
        uint32_t message = ((sampleData << ID_BITS) | messageId);
        message_d[warpLane] = message;
        __threadfence(); 
    }
    if(warpIsInTheHierarchy) {
        // Receiving the message and sending it to the next two warps, with minimal contention.
        uint32_t receivedSampleData = d_broadcast(messageId, warpLane, message_d, numCommunicatingWarps, globalWarpId);
        // Writing the result.
        output_d[globalThreadId] = receivedSampleData;
    }
}