#include "lcg_kernel.h"

__global__ void k_calculateLCG(uint32_t * __restrict__ data_d, int numElements) {
    constexpr int EXTRA_ITERATIONS_FOR_DOUBLE_BUFFERING_PIPELINE = 2;
    constexpr int STARTING_ITERATION_FOR_COMPUTATION = 1;
    int numThreadsPerBlock = blockDim.x;
    int globalThreadId = threadIdx.x + blockIdx.x * numThreadsPerBlock;
    int numTotalThreads = numThreadsPerBlock * gridDim.x;
    int numElementsPerGrid = numTotalThreads * MEMORY_ACCESS_WIDTH;
    int numGridStrideIterations = EXTRA_ITERATIONS_FOR_DOUBLE_BUFFERING_PIPELINE + 1 + (numElements - 1) / numElementsPerGrid;
    // Mapping the dynamically allocated shared memory to two regions to be used asynchronously.
    extern __shared__ char s_dynamicBuffer[];
    uint4 * __restrict__ s_asyncLoadStorage1 = reinterpret_cast<uint4*>(&s_dynamicBuffer[0]);
    uint4 * __restrict__ s_asyncLoadStorage2 = s_asyncLoadStorage1 + numThreadsPerBlock;
    int storageSelector = 0;
    for(int stride = 0; stride < numGridStrideIterations; stride++) {
        int id = stride * numElementsPerGrid + globalThreadId * MEMORY_ACCESS_WIDTH;
        auto selectedStorage = storageSelector++ & 1;
        // Asynchronously loading the next iteration's data.
        if(stride < numGridStrideIterations - EXTRA_ITERATIONS_FOR_DOUBLE_BUFFERING_PIPELINE){
            if(id < numElements) {
                // Reading multiple elements at once.
                uint4 * __restrict__ accessPtr = reinterpret_cast<uint4*>(&data_d[id]);
                __pipeline_memcpy_async(selectedStorage ? &s_asyncLoadStorage1[threadIdx.x] : &s_asyncLoadStorage2[threadIdx.x], accessPtr, sizeof(uint4));
                __pipeline_commit();
            }
        }
        // The element index for computation is based on the async loading index, lagging by numElementsPerGrid elements.
        int idForComputation = id - numElementsPerGrid;
        if(idForComputation < numElements && stride >= STARTING_ITERATION_FOR_COMPUTATION) {
            // Reading multiple elements at once.
            uint4 * __restrict__ accessPtr = reinterpret_cast<uint4*>(selectedStorage ? &s_asyncLoadStorage2[threadIdx.x] : &s_asyncLoadStorage1[threadIdx.x]);
            // Computing the linear congruential generator algorithm for each 32bit element, one at a time to lower the register pressure for higher occupancy and hiding more latency.
            uint64_t data = accessPtr->x;
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            accessPtr->x = data;
            data = accessPtr->y;
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            accessPtr->y = data;
            data = accessPtr->z;
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            accessPtr->z = data;
            data = accessPtr->w;
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            accessPtr->w = data;
            // Writing multiple elements at once.
            *reinterpret_cast<uint4*>(&data_d[idForComputation]) = *accessPtr;
        }
        __pipeline_wait_prior(0);
        __syncthreads();
    }
}