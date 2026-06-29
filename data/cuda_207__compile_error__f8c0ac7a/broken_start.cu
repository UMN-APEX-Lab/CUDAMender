#include <cstdint>

__global__ void k_calculateElement(uint32_t booleanElementsPerInteger, uint32_t numIntegers, uint32_t* integersIn_d, uint32_t* integersOut_d) {
    // Each thread processes 128 booleans = 4 uint32_t = 1 uint4
    const uint32_t booleansPerThread = 128;
    const uint32_t intsPerThread = booleansPerThread / 32; // 4
    
    // Total number of booleans in the array
    const uint32_t totalBooleans = numIntegers * booleanElementsPerInteger;
    
    // Global thread index
    const uint32_t threadIdxGlobal = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Starting boolean index for this thread
    const uint32_t startBooleanIdx = threadIdxGlobal * booleansPerThread;
    
    // If this thread has no work to do, return
    if (startBooleanIdx >= totalBooleans) {
        return;
    }
    
    // Load 4 uint32_t values as uint4 (vectorized load)
    uint4 data;
    uint32_t* inputPtr = integersIn_d;
    
    // Calculate which uint32_t indices we need to read
    // We need to handle boundaries carefully since we need neighbors
    
    // For 128 booleans starting at startBooleanIdx, we need to cover
    // booleans from startBooleanIdx-1 to startBooleanIdx+128 (inclusive for XOR window)
    // Actually for XOR of 3 consecutive bits, output bit i depends on bits i-1, i, i+1
    
    // We'll process 128 output booleans, but need to read 130 input booleans (with padding)
    
    // First, let's read the main 4 uint32_t values
    uint32_t baseIntIdx = startBooleanIdx / 32;
    
    // We need to handle the fact that we might read out of bounds
    // For each output boolean at position i, we need input booleans at i-1, i, i+1
    
    uint32_t vals[6]; // Extra space for boundary handling: need 4+2 = 6 uint32_t potentially
    
    // Read the 4 uint32_t values for this thread's output, plus boundary values
    for (int i = -1; i <= 4; i++) {
        int readIdx = baseIntIdx + i;
        if (readIdx >= 0 && readIdx < (int)numIntegers) {
            vals[i + 1] = inputPtr[readIdx];
        } else {
            vals[i + 1] = 0;
        }
    }
    
    // Now compute 128 output bits
    // Each output bit i (0-127) corresponds to global boolean index startBooleanIdx + i
    // It needs input bits at (startBooleanIdx + i - 1), (startBooleanIdx + i), (startBooleanIdx + i + 1)
    
    uint32_t outVals[4] = {0, 0, 0, 0};
    
    for (int i = 0; i < 128; i++) {
        uint32_t globalBitIdx = startBooleanIdx + i;
        
        // The three input bit positions (in terms of global boolean index)
        // We need bits at globalBitIdx-1, globalBitIdx, globalBitIdx+1
        
        // Calculate which vals[] entry and which bit within that entry
        // vals[0] corresponds to baseIntIdx-1, i.e., booleans [(baseIntIdx-1)*32, (baseIntIdx-1)*32+31]
        // vals[1] corresponds to baseIntIdx, etc.
        
        // Relative to baseIntIdx*32 (start of vals[1])
        int32_t relBitIdx = (int32_t)globalBitIdx - (int32_t)(baseIntIdx * 32);
        
        // Get the three bits
        uint32_t bit0 = 0, bit1 = 0, bit2 = 0;
        
        // Bit at position relBitIdx - 1 (in vals array space)
        int32_t pos0 = relBitIdx - 1;
        if (pos0 >= -32 && pos0 < (int32_t)(5 * 32)) {
            int valIdx = (pos0 + 32) / 32; // which vals[] entry
            int bitPos = (pos0 + 32) % 32; // which bit within that entry
            if (valIdx >= 0 && valIdx < 6) {
                bit0 = (vals[valIdx] >> bitPos) & 1;
            }
        }
        
        // Bit at position relBitIdx
        int32_t pos1 = relBitIdx;
        if (pos1 >= -32 && pos1 < (int32_t)(5 * 32)) {
            int valIdx = (pos1 + 32) / 32;
            int bitPos = (pos1 + 32) % 32;
            if (valIdx >= 0 && valIdx < 6) {
                bit1 = (vals[valIdx] >> bitPos) & 1;
            }
        }
        
        // Bit at position relBitIdx + 1
        int32_t pos2 = relBitIdx + 1;
        if (pos2 >= -32 && pos2 < (int32_t)(5 * 32)) {
            int valIdx = (pos2 + 32) / 32;
            int bitPos = (pos2 + 32) % 32;
            if (valIdx >= 0 && valIdx < 6) {
                bit2 = (vals[valIdx] >> bitPos) & 1;
            }
        }
        
        // XOR of three bits
        uint32_t result = bit0 ^ bit1 ^ bit2;
        
        // Store in appropriate output uint32_t
        int outIntIdx = i / 32;
        int outBitPos = i % 32;
        outVals[outIntIdx] |= (result << outBitPos);
    }
    
    // Write results using uint4 vectorized store
    uint4 outputData;
    outputData.x = outVals[0];
    outputData.y = outVals[1];
    outputData.z = outVals[2];
    outputData.w = outVals[3];
    
    // Check bounds before writing
    if (baseIntIdx + 3 < numIntegers) {
        // Can write all 4
        reinterpret_cast<uint4*>(integersOut_d + baseIntIdx)[0] = outputData;
    } else {
        // Write individually with bounds checking
        for (int i = 0; i < 4; i++) {
            if (baseIntIdx + i < numIntegers) {
                integersOut_d[baseIntIdx + i] = outVals[i];
            }
        }
    }
}