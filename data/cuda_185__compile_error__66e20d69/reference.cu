#include "log2_kernel.h"

__global__ void k_calculateLog2(int numElements, float * data_d, float * lookupTable_d, int lookupTableDuplication) {
    constexpr int NUM_MANTISSA_BITS = 23;
    constexpr int EXPONENT_BIAS = 127;
    constexpr int EXPONENT_MASK = 0b11111111;
    constexpr int MANTISSA_MASK = 0b11111111111111111111111;
    extern __shared__ char s_mem[];
    float * s_lookup = reinterpret_cast<float*>(s_mem);
    // Initializing the duplicate lookup table elements without bank conflicts..
    for(int workIndex = threadIdx.x; workIndex < LOOKUP_TABLE_ELEMENTS; workIndex += blockDim.x) {
        float lookupValue = lookupTable_d[workIndex];
        for(int i = 0; i < lookupTableDuplication; i++) {
            s_lookup[workIndex * lookupTableDuplication + ((workIndex + i) % lookupTableDuplication)] = lookupValue;
        }
    }
    __syncthreads();
    int globalThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int numGlobalThreads = blockDim.x * gridDim.x;
    for(int workIndex = globalThreadIndex; workIndex < numElements; workIndex += numGlobalThreads) {
        // Computing log2(x) = (exponent bits of x) + log2(mantissa bits of x), where the log2(mantissa) is fetched from the lookup table.
        // Coalesced access to global memory.
        float x = data_d[workIndex];
        int bitRepresentation = __float_as_int(x);
        int exponentBitRepresentation = (bitRepresentation >> NUM_MANTISSA_BITS) & EXPONENT_MASK;
        int exponent = exponentBitRepresentation - EXPONENT_BIAS;
        float mantissa = __int_as_float((EXPONENT_BIAS << NUM_MANTISSA_BITS) | (bitRepresentation & MANTISSA_MASK));
        int lookupIndex = int((mantissa - 1.0f) * LOOKUP_TABLE_ELEMENTS);
        // Calculating indices that are free from bank conflicts to utilize more memory banks concurrently.
        float result = exponent + s_lookup[lookupIndex * lookupTableDuplication + (threadIdx.x % lookupTableDuplication)];
        // Coalesced access to global memory.
        data_d[workIndex] = result;
    }
}