#include "lbp.h"

__global__ void k_computeLBP(const unsigned char* input_d, unsigned char* output_d, int width, int height) {
    // Shared memory for tile with halo (padding)
    extern __shared__ unsigned char s_tileData[];

    const int paddedTileWidth = blockDim.x + 2;
    const int paddedTileHeight = blockDim.y + 2;
    const int totalTileElements = paddedTileWidth * paddedTileHeight;

    // Grid-stride loop for processing multiple tiles
    for (int currentBlockY = blockIdx.y; currentBlockY * blockDim.y < height; currentBlockY += gridDim.y) {
        for (int currentBlockX = blockIdx.x; currentBlockX * blockDim.x < width; currentBlockX += gridDim.x) {

            // Calculate block origin coordinates
            int blockOriginX = currentBlockX * blockDim.x;
            int blockOriginY = currentBlockY * blockDim.y;

            // Thread coordination for cooperative loading
            int totalThreadsInBlock = blockDim.x * blockDim.y;
            int flatThreadIndex = threadIdx.y * blockDim.x + threadIdx.x;

            // Cooperative loading of tile data (including halo region)
            for (int elementIndex = flatThreadIndex; elementIndex < totalTileElements; elementIndex += totalThreadsInBlock) {
                int tileRow = elementIndex / paddedTileWidth;
                int tileCol = elementIndex % paddedTileWidth;

                // Convert tile coordinates to global coordinates (subtract 1 for halo)
                int globalX = blockOriginX + tileCol - 1;
                int globalY = blockOriginY + tileRow - 1;

                // Clamp coordinates to handle image boundaries
                int clampedX = max(0, min(globalX, width - 1));
                int clampedY = max(0, min(globalY, height - 1));

                s_tileData[elementIndex] = input_d[clampedY * width + clampedX];
            }

            __syncthreads();

            // Process all pixels in the current block
            int outputX = blockOriginX + threadIdx.x;
            int outputY = blockOriginY + threadIdx.y;

            if (outputX < width && outputY < height) {
                // Initialize boundary pixels to 0
                if (outputX == 0 || outputX == width - 1 || outputY == 0 || outputY == height - 1) {
                    output_d[outputY * width + outputX] = 0;
                } else {
                    // Compute LBP for interior pixels
                    int sharedX = threadIdx.x + 1;
                    int sharedY = threadIdx.y + 1;
                    int centerPixelIndex = sharedY * paddedTileWidth + sharedX;

                    unsigned char centerPixelValue = s_tileData[centerPixelIndex];

                    // Compute LBP code using 8-neighbor pattern (clockwise from top-left)
                    unsigned char lbpCode = 0;
                    lbpCode |= (s_tileData[centerPixelIndex - paddedTileWidth - 1] >= centerPixelValue) << BIT_TOP_LEFT;     // top-left
                    lbpCode |= (s_tileData[centerPixelIndex - paddedTileWidth]     >= centerPixelValue) << BIT_TOP;          // top
                    lbpCode |= (s_tileData[centerPixelIndex - paddedTileWidth + 1] >= centerPixelValue) << BIT_TOP_RIGHT;    // top-right
                    lbpCode |= (s_tileData[centerPixelIndex + 1]                   >= centerPixelValue) << BIT_RIGHT;        // right
                    lbpCode |= (s_tileData[centerPixelIndex + paddedTileWidth + 1] >= centerPixelValue) << BIT_BOTTOM_RIGHT; // bottom-right
                    lbpCode |= (s_tileData[centerPixelIndex + paddedTileWidth]     >= centerPixelValue) << BIT_BOTTOM;       // bottom
                    lbpCode |= (s_tileData[centerPixelIndex + paddedTileWidth - 1] >= centerPixelValue) << BIT_BOTTOM_LEFT;  // bottom-left
                    lbpCode |= (s_tileData[centerPixelIndex - 1]                   >= centerPixelValue) << BIT_LEFT;         // left

                    output_d[outputY * width + outputX] = lbpCode;
                }
            }
            __syncthreads();
        }
    }
}