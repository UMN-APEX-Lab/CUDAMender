#include "kernel_contract.h"

__global__ void k_calculateDifferencesFromPowerOfAveragePerChunk(   int32_t * input_d, 
                                                                    float * output_d, 
                                                                    int numChunks) {
    // Each thread index will compute a different set of four chunks for writing coalesced output.
    int globalThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Grid-stride loop that supports any number of chunks.
    int numGlobalThreads = blockDim.x * gridDim.x;
    int numIterations = (numChunks + numGlobalThreads - 1) / numGlobalThreads;
    for(int iteration = 0; iteration < numIterations; iteration++) {
        int chunkId = numGlobalThreads * iteration + globalThreadIndex;
        // Processing data from the current iteration.
        if(chunkId < numChunks) {
            // Reading inputs in a coalesced form to hide latency.
            // Loading four consecutive integers as a single int4 data.
            int4 data = *reinterpret_cast<int4*>(&input_d[chunkId * CHUNK_ELEMENTS]);
            // Computing.
            float average = data.x * 0.25f + data.y * 0.25f + data.z * 0.25f + data.w * 0.25f;
            // Finding highest bit count.
            int bitCount = __popc(data.x) > __popc(data.y) ? __popc(data.x) : __popc(data.y);
            bitCount = (bitCount > __popc(data.z) ? bitCount : __popc(data.z));
            bitCount = (bitCount > __popc(data.w) ? bitCount : __popc(data.w));
            float4 result = { 
                data.x - powf(average, (float)bitCount),
                data.y - powf(average, (float)bitCount),
                data.z - powf(average, (float)bitCount),
                data.w - powf(average, (float)bitCount)
            };
            // Writing results in a coalesced form to minimize latency.
            *reinterpret_cast<float4*>(&output_d[chunkId * CHUNK_ELEMENTS]) = result; 
        }
    }
}