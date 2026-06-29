#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#ifndef CHUNK_ELEMENTS
#define CHUNK_ELEMENTS 4
#endif

#ifndef INPUT_CHUNK_TYPE
struct int4 {
    int x, y, z, w;
};
#define INPUT_CHUNK_TYPE int4
#endif

__device__ inline int popcount(int x) {
    return __popc(x);
}

__global__ void k_calculateDifferencesFromPowerOfAveragePerChunk(
    int32_t * input_d,
    float * output_d,
    int numChunks) {
    
    int chunkIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (chunkIdx >= numChunks) return;
    
    // Coalesced load using int4
    INPUT_CHUNK_TYPE chunk = reinterpret_cast<INPUT_CHUNK_TYPE*>(input_d)[chunkIdx];
    
    int vals[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
    
    // Find the number with highest bit count
    int maxBits = -1;
    int maxBitsIdx = 0;
    for (int i = 0; i < 4; i++) {
        int bits = popcount(vals[i]);
        if (bits > maxBits) {
            maxBits = bits;
            maxBitsIdx = i;
        }
    }
    
    int n = maxBits;
    
    // Compute average of the four numbers
    float sum = 0.0f;
    for (int i = 0; i < 4; i++) {
        sum += static_cast<float>(vals[i]);
    }
    float a = sum / 4.0f;
    
    // Compute a^n
    float a_pow_n = powf(a, static_cast<float>(n));
    
    // Compute output values and store coalesced
    float4 output;
    output.x = static_cast<float>(vals[0]) - a_pow_n;
    output.y = static_cast<float>(vals[1]) - a_pow_n;
    output.z = static_cast<float>(vals[2]) - a_pow_n;
    output.w = static_cast<float>(vals[3]) - a_pow_n;
    
    reinterpret_cast<float4*>(output_d)[chunkIdx] = output;
}