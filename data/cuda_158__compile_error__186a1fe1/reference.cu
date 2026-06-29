#include "lbp_kernel.h"
#include <cmath>
#include <algorithm>

__global__ void k_localBinaryKernel(const unsigned char* input, unsigned char* output, int width, int segHeight, int numNeighbors, float radius) {
    // Block dimensions
    int bx = blockDim.x; 
    int by = blockDim.y;

    // Compute the shared memory tile dimensions
    const int tileWidth = bx + 2;
    const int tileHeight = by + 2;

    // Dynamic shared memory
    extern __shared__ unsigned char s_data[];

    int baseRow = blockIdx.y * by;
    int baseCol = blockIdx.x * bx;
    for (int row = baseRow; row < segHeight; row += gridDim.y * by) {
        for (int col = baseCol; col < width; col += gridDim.x * bx) {

            // The shared tile has totalTileElements
            int totalTileElements = tileWidth * tileHeight;

            // Compute flat thread index within the block
            int flatIdx = threadIdx.y * bx + threadIdx.x;
            int blockSize2D = bx * by;

            // Each thread cooperatively loads one or more elements from the tile
            for (int idx = flatIdx; idx < totalTileElements; idx += blockSize2D) {

                // row in tile
                int tileRow = idx / tileWidth;
                // col in tile 
                int tileCol = idx % tileWidth;
                
                // subtract 1 for halo offset
                int globalRow = row + tileRow - 1; 
                int globalCol = col + tileCol - 1;
                if (globalRow < 0 || globalRow >= segHeight || globalCol < 0 || globalCol >= width)
                    s_data[idx] = 0;
                else
                    s_data[idx] = input[globalRow * width + globalCol];
            }
            // Synchronize
            __syncthreads();

            // Each thread computes the LBP for the pixel at threadIdx.y, threadIdx.x 
            // within the interior of the tile.
            int xCord = threadIdx.x + 1;
            int yCord = threadIdx.y + 1;
            int globalPixelRow = row + threadIdx.y;
            int globalPixelCol = col + threadIdx.x;
            if (globalPixelRow < segHeight && globalPixelCol < width){
                unsigned char center = s_data[yCord * tileWidth + xCord];
                unsigned char lbp = 0;
                const float pi = 3.14159265358979323846f;

                // Loop over each neighbor
                for (int k = 0; k < numNeighbors; ++k){
                    float angle = 2.0f * pi * k / numNeighbors;

                    // Compute the neighbor coordinates in shared memory space.
                    float sampleX = xCord + radius * cosf(angle);
                    float sampleY = yCord - radius * sinf(angle);

                    //Lower and upper co-ordinates
                    int xCord0 = (int)floorf(sampleX);
                    int yCord0 = (int)floorf(sampleY);
                    int xCord1 = xCord0 + 1;
                    int yCord1 = yCord0 + 1;
                    float wx = sampleX - xCord0;
                    float wy = sampleY - yCord0;

                    // Clamp coordinates to valid range within the segment
                    xCord0 = max(0, min(xCord0, tileWidth - 1));
                    xCord1 = max(0, min(xCord1, tileWidth - 1));
                    yCord0 = max(0, min(yCord0, tileHeight - 1));
                    yCord1 = max(0, min(yCord1, tileHeight - 1));

                    // Fetch the four surrounding pixel values
                    // Top-left pixel
                    float value00 = (float)s_data[yCord0 * tileWidth + xCord0];

                    // Top-right pixel
                    float value01 = (float)s_data[yCord0 * tileWidth + xCord1];

                    // Bottom-left pixel
                    float value10 = (float)s_data[yCord1 * tileWidth + xCord0];

                    // Bottom-right pixel
                    float value11 = (float)s_data[yCord1 * tileWidth + xCord1];

                    // Compute the interpolated intensity
                    float interpVal = (1 - wx) * (1 - wy) * value00 + wx * (1 - wy) * value01 + (1 - wx) * wy * value10 + wx * wy * value11;

                    // Compare the interpolated neighbor intensity with the center and set the corresponding bit
                    lbp |= ((interpVal >= center) ? 1 : 0) << (numNeighbors - 1 - k);
                }
                // Write the computed LBP code to the output array
                output[globalPixelRow * width + globalPixelCol] = lbp;
            }
            // Synchronize
            __syncthreads();
        }
    }
}

// Dynamic shared memory size
__host__ size_t dynamicSMemSizeFunc(int blockSize) {
    int blockDimX = min(MAX_BLOCKS_PER_SEGMENT, blockSize); 
    int blockDimY = max(1, blockSize / blockDimX);
    
    // Calculate the tile dimensions with halo
    int tileWidth = blockDimX + 2;
    int tileHeight = blockDimY + 2;
    
    return tileWidth * tileHeight * sizeof(unsigned char);
}

// Determines optimal block and grid sizes using occupancy APIs.
cudaError_t getOptimalLaunchParams(int segmentSize, int &optBlockSize, int &blocksPerGrid, float &theoreticalOccupancy) {
    size_t dynamicSharedMemSize = 0;

    int minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
        &minGridSize, &optBlockSize, k_localBinaryKernel,
        dynamicSMemSizeFunc, dynamicSharedMemSize, cudaOccupancyPreferShared));

    // Query max active blocks per SM
    size_t requiredSMem = dynamicSMemSizeFunc(optBlockSize);
    int maxActiveBlocksPerSM;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocksPerSM, k_localBinaryKernel, optBlockSize, requiredSMem));

    size_t availableDynamicSharedMemory;
    CUDA_CHECK(cudaOccupancyAvailableDynamicSMemPerBlock(
        &availableDynamicSharedMemory,
        (const void*)k_localBinaryKernel,
        maxActiveBlocksPerSM,
        optBlockSize));

    if (requiredSMem > availableDynamicSharedMemory) {
        // If required shared memory is more than available, reduce block size.
        int maxElementsInSharedMem = availableDynamicSharedMemory / sizeof(unsigned char);
        // Calculate new block dimensions that would fit in available shared memory
        int blockDimX = min(MAX_BLOCKS_PER_SEGMENT, optBlockSize);
        int blockDimY = max(1, optBlockSize / blockDimX);
       
        // Calculate new tile dimensions 
        int tileWidth = blockDimX + 2;
        int tileHeight = blockDimY + 2;
       
        // Calculate the maximum block size 
        // We need to maintain the original thread layout while reducing total threads
        float scaleFactor = sqrtf(maxElementsInSharedMem / (float)(tileWidth * tileHeight));
        blockDimX = min(MAX_BLOCKS_PER_SEGMENT, (int)(blockDimX * scaleFactor));
        blockDimY = max(1, (int)(blockDimY * scaleFactor));
       
        // Update the optimal block size
        optBlockSize = blockDimX * blockDimY;
    }

    // Query device properties.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    // Compute maximum active blocks per multiprocessor.
    int maxActiveBlocks;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, k_localBinaryKernel, optBlockSize, requiredSMem));
    theoreticalOccupancy = (maxActiveBlocks * optBlockSize) / (float)prop.maxThreadsPerMultiProcessor;

    blocksPerGrid = (segmentSize + optBlockSize - 1) / optBlockSize;
    blocksPerGrid = std::min(blocksPerGrid, MAX_BLOCKS_PER_SEGMENT);
    return cudaSuccess;
}