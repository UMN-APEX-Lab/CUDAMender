#include "log2_kernel.h"

__global__ void k_calculateLog2(int numElements, float *data_d, float *lookupTable_d, int lookupTableDuplication) {
    const int LOOKUP_TABLE_ELEMENTS = 256;
    
    extern __shared__ float sharedLookupTable[];
    
    int tid = threadIdx.x;
    int totalSharedElements = LOOKUP_TABLE_ELEMENTS * lookupTableDuplication;
    
    // Load lookup table from global memory to shared memory with coalesced access
    for (int i = tid; i < totalSharedElements; i += blockDim.x) {
        int originalIndex = i / lookupTableDuplication;
        sharedLookupTable[i] = lookupTable_d[originalIndex];
    }
    
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numElements) {
        float x = data_d[idx];
        
        // Extract exponent and mantissa from IEEE 754 float
        int xInt = __float_as_int(x);
        
        // Extract exponent (bits 23-30)
        int exponent = ((xInt >> 23) & 0xFF) - 127;
        
        // Extract mantissa (bits 0-22) - 23 bits
        int mantissa = xInt & 0x007FFFFF;
        
        // Compute lookup table index from upper bits of mantissa
        int lookupIndex = mantissa >> (23 - 8); // Get upper 8 bits of mantissa
        
        // Clamp lookup index to valid range
        if (lookupIndex >= LOOKUP_TABLE_ELEMENTS) lookupIndex = LOOKUP_TABLE_ELEMENTS - 1;
        if (lookupIndex < 0) lookupIndex = 0;
        
        // Compute bank-aware index to avoid bank conflicts
        // Use threadIdx.x % lookupTableDuplication to select which duplicate
        int bankOffset = threadIdx.x % lookupTableDuplication;
        int sharedIndex = lookupIndex * lookupTableDuplication + bankOffset;
        
        float lookupValue = sharedLookupTable[sharedIndex];
        
        // Reconstruct log2(x) from exponent and lookup value
        // log2(x) = exponent + log2(1 + mantissa_fraction)
        // The lookup table stores log2(1 + index/256)
        // We need to interpolate or use the lookup value directly
        
        // For more accuracy, we can use the formula:
        // log2(x) = exponent + log2(1.f + mantissa / (float)(1 << 23))
        // The lookup table gives us log2(1 + lookupIndex/256)
        // But we need more precision, so we use the fractional part
        
        float mantissaFrac = mantissa / (float)(1 << 23);
        float fineFrac = (lookupIndex / 256.0f);
        float remainder = mantissaFrac - fineFrac;
        
        // Linear interpolation for better accuracy
        float nextLookupValue = (lookupIndex + 1 < LOOKUP_TABLE_ELEMENTS) ? 
            sharedLookupTable[(lookupIndex + 1) * lookupTableDuplication + bankOffset] : 
            lookupValue;
        
        float frac = (mantissaFrac * 256.0f) - lookupIndex;
        float interpolatedValue = lookupValue + frac * (nextLookupValue - lookupValue);
        
        float result = exponent + interpolatedValue;
        
        data_d[idx] = result;
    }
}