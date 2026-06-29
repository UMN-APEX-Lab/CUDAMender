#include "miller_rabin.h"

int gWarpSize;

// Compute the required dynamic shared memory size for k_millerRabin.
__host__ size_t dynamicSMemSizeMillerRabin(int blockSize) {
    int numWarps = blockSize / gWarpSize;
    return numWarps * (sizeof(unsigned long long) + sizeof(int));
}

// Determines the optimal launch parameters using CUDA occupancy APIs.
cudaError_t getOptimalLaunchParamsMillerRabin(int numTests, int &optBlockSize, int &blocksPerGrid, float &theoreticalOccupancy) {
    size_t dynamicSharedMemSize = 0;
    int minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(&minGridSize, &optBlockSize,
                     k_millerRabin, dynamicSMemSizeMillerRabin, dynamicSharedMemSize, cudaOccupancyPreferShared));

    int maxActiveBlocksPerSM;
    size_t requiredSMem = dynamicSMemSizeMillerRabin(optBlockSize);
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM,
                     k_millerRabin, optBlockSize, requiredSMem));

    size_t availableDynamicSharedMemory;
    CUDA_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock(&availableDynamicSharedMemory,
                     (const void*)k_millerRabin, maxActiveBlocksPerSM, optBlockSize));

    if (requiredSMem > availableDynamicSharedMemory) {
        optBlockSize = availableDynamicSharedMemory / (sizeof(unsigned long long) + sizeof(int));
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    theoreticalOccupancy = (maxActiveBlocksPerSM * optBlockSize) / (float)prop.maxThreadsPerMultiProcessor;

    blocksPerGrid = (numTests + optBlockSize - 1) / optBlockSize;
    blocksPerGrid = std::min(blocksPerGrid, MAX_BLOCKS_PER_SEGMENT);
    return cudaSuccess;
}