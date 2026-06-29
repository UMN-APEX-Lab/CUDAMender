#include "broadcast_tree.h"
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t d_broadcast(uint32_t messageId,
                                                int warpLane,
                                                uint32_t * message_d,
                                                int numCommunicatingWarps,
                                                int globalWarpId) {
    // Calculate the index for this warp's message location
    int index = globalWarpId * warpSize + warpLane;
    uint32_t expectedMessage = (messageId & 0xFF) | (0xFFFFFF00 & ~0); // Full mask for expected, but we check properly
    
    uint32_t message = 0;
    
    // Wait for message with matching ID using atomic exchange
    // Keep trying until we get a non-zero message with matching ID
    while (true) {
        // Atomic exchange: read and set to 0
        message = atomicExch(&message_d[index], 0);
        
        if (message != 0) {
            uint32_t receivedId = message & 0xFF;
            if (receivedId == messageId) {
                // Found our message
                break;
            } else {
                // Wrong message ID, put it back and wait
                // This shouldn't happen in correct tree broadcast, but handle it
                uint32_t old = atomicCAS(&message_d[index], 0, message);
                // If someone else wrote, we lost our message - rare race
                // Just continue waiting
            }
        }
        // Memory fence to ensure visibility
        __threadfence();
    }
    
    // Now forward to children if they exist
    // In complete binary tree, node k has children at 2*k+1 and 2*k+2
    int leftChild = 2 * globalWarpId + 1;
    int rightChild = 2 * globalWarpId + 2;
    
    // Extract the 24-bit data from the message
    uint32_t data = message >> 8;
    
    // Prepare message for children (same data, same ID)
    uint32_t childMessage = (data << 8) | (messageId & 0xFF);
    
    // Forward to left child if it exists
    if (leftChild < numCommunicatingWarps) {
        int leftIndex = leftChild * warpSize + warpLane;
        // Wait until location is 0 (ready to receive), then write
        while (atomicCAS(&message_d[leftIndex], 0, childMessage) != 0) {
            __threadfence();
        }
    }
    
    // Forward to right child if it exists
    if (rightChild < numCommunicatingWarps) {
        int rightIndex = rightChild * warpSize + warpLane;
        // Wait until location is 0 (ready to receive), then write
        while (atomicCAS(&message_d[rightIndex], 0, childMessage) != 0) {
            __threadfence();
        }
    }
    
    // Memory fence to ensure writes are visible
    __threadfence();
    
    // Return the 24-bit data
    return data;
}

__global__ void k_broadcastWithHierarchicalPath(uint32_t input,
                                                uint32_t * message_d,
                                                int numCommunicatingWarps,
                                                int * output_d) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpLane = tid % warpSize;
    int globalWarpId = tid / warpSize;
    
    // Only participating warps should do work
    if (globalWarpId >= numCommunicatingWarps) {
        return;
    }
    
    uint32_t messageId = (uint32_t)(warpLane + 1); // 1-based lane index
    uint32_t data;
    
    if (globalWarpId == 0) {
        // First warp: compute message and write to buffer for itself, then broadcast
        // Compute: input + warpLane (0-based)
        data = input + warpLane;
        
        // Encode: 24-bit data in MSB, 8-bit ID in LSB
        uint32_t message = (data << 8) | (messageId & 0xFF);
        
        // Write to our own location (index 0 * warpSize + warpLane = warpLane)
        // But we need to write to child locations, not our own
        // Actually, warp 0 should write to its children, and also "receive" from itself
        
        // For warp 0, we write to children directly
        int leftChild = 1;  // 2*0+1
        int rightChild = 2; // 2*0+2
        
        // Write to left child if exists
        if (leftChild < numCommunicatingWarps) {
            int leftIndex = leftChild * warpSize + warpLane;
            // Ensure location is 0 first
            while (atomicCAS(&message_d[leftIndex], 0, message) != 0) {
                __threadfence();
            }
        }
        
        // Write to right child if exists
        if (rightChild < numCommunicatingWarps) {
            int rightIndex = rightChild * warpSize + warpLane;
            while (atomicCAS(&message_d[rightIndex], 0, message) != 0) {
                __threadfence();
            }
        }
        
        __threadfence();
        
        // Warp 0 doesn't need to receive, it already has the data
        // But we need to follow the protocol - actually, let's use d_broadcast for consistency
        // Actually, re-reading: "All warps, including warp 0 which initiates the broadcast, 
        // should use the d_broadcast function to read their message from the message buffer."
        // So warp 0 needs to write to its own location and then call d_broadcast
        
        // Wait, the description says warp 0 writes to message buffer before calling d_broadcast
        // But d_broadcast reads from its own location. So warp 0 should write to location 0
        
        // Let me re-implement: warp 0 writes to its own slot, then calls d_broadcast
        // But that would cause a deadlock since d_broadcast waits for non-zero
        
        // Actually, re-reading: "Warp 0 prepares and writes its computed message to the message buffer 
        // before proceeding with the broadcast protocol."
        
        // The issue is d_broadcast expects to read from its own location. For warp 0 to use
        // d_broadcast, it needs to write to location 0 first. But then atomicExch would get it.
        
        // However, looking at the tree structure, if we follow strictly:
        // - Each node receives from parent and forwards to children
        // - Root has no parent, so it should just forward to children
        
        // But the requirement says ALL warps use d_broadcast. So let's make warp 0
        // write to its own location, then use a special path or modify logic.
        
        // Actually, simpler: let's have warp 0 skip the wait loop by pre-populating,
        // but still call d_broadcast for code uniformity. We can make d_broadcast
        // handle the case where the location already has the right message.
        
        // For now, let's write to our own location and use atomicExch to read it
        int myIndex = globalWarpId * warpSize + warpLane;
        message_d[myIndex] = message; // Non-atomic write, we'll atomicExch it
        
        __threadfence();
        
        // Now call d_broadcast which will read this
        data = d_broadcast(messageId, warpLane, message_d, numCommunicatingWarps, globalWarpId);
    } else {
        // Other warps: just call d_broadcast to receive and forward
        data = d_broadcast(messageId, warpLane, message_d, numCommunicatingWarps, globalWarpId);
    }
    
    // Write output
    int outputIndex = globalWarpId * warpSize + warpLane;
    output_d[outputIndex] = (int)data;
}