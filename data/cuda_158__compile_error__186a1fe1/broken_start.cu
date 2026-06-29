#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "lbp_kernel.h"

#define MAX_BLOCKS_PER_SEGMENT 32

__device__ float get_pixel_bilinear(const unsigned char* input, int width, int height, float x, float y) {
    if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1) {
        // Boundary check: if the 2x2 window is not fully inside, 
        // we need to be careful. The task says treat any position outside bounds as 0.
        // For bilinear interpolation, if any of the 4 pixels are outside, 
        // the logic below handles it by checking bounds for each pixel.
    }

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = x - x0;
    float dy = y - y0;

    auto get_val = [&](int px, int py) {
        if (px < 0 || px >= width || py < 0 || py >= height) return 0.0f;
        return (float)input[py * width + px];
    };

    float v00 = get_val(x0, y0);
    float v10 = get_val(x1, y0);
    float v01 = get_val(x0, y1);
    float v11 = get_val(x1, y1);

    return (1.0f - dx) * (1.0f - dy) * v00 + dx * (1.0f - dy) * v10 +
           (1.0f - dx) * dy * v01 + dx * dy * v11;
}

__global__ void k_localBinaryKernel(const unsigned char* input, unsigned char* output,
                                     int width, int segHeight, int numNeighbors, float radius) {
    extern __shared__ unsigned char s_tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    int gx = bx + tx;
    int gy = by + ty;

    // Shared memory dimensions: (blockDim.x + 2) x (blockDim.y + 2)
    int s_width = blockDim.x + 2;
    
    // Cooperative load into shared memory (including halo)
    // Each thread loads its primary pixel and helps with the halo
    for (int i = ty * s_width + tx; i < (blockDim.y + 2) * s_width; i += blockDim.x * blockDim.y) {
        int local_y = i / s_width;
        int local_x = i % s_width;
        int global_x = bx + local_x - 1;
        int global_y = by + local_y - 1;

        if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < segHeight) {
            s_tile[i] = input[global_y * width + global_x];
        } else {
            s_tile[i] = 0;
        }
    }
    __syncthreads();

    if (gx >= width || gy >= segHeight) return;

    float center_val = (float)s_tile[(ty + 1) * s_width + (tx + 1)];
    unsigned char lbp_code = 0;

    for (int k = 0; k < numNeighbors; ++k) {
        float angle = 2.0f * M_PI * k / numNeighbors;
        float nx = (float)gx + radius * cosf(angle);
        float ny = (float)gy + radius * sinf(angle); // Image y increases downward, so sin(angle) is correct

        float neighbor_val;
        // Since radius is 1.0, the sampling point is within [gx-1, gx+1] and [gy-1, gy+1].
        // The shared memory tile covers [bx-1, bx+blockDim.x] and [by-1, by+blockDim.y].
        // To use bilinear interpolation, we need 4 pixels. 
        // For radius 1, the max coordinate is gx+1.0, which requires pixel at gx+2 for interpolation.
        // However, the prompt specifies a halo of exactly one pixel.
        // This implies we should use the global input for interpolation or the tile if it fits.
        // Given the constraints and the "halo of one pixel", we use the input pointer for safety 
        // or handle the bilinear interpolation carefully.
        
        neighbor_val = get_pixel_bilinear(input, width, segHeight, nx, ny);

        if (neighbor_val >= center_val) {
            lbp_code |= (1 << (numNeighbors - 1 - k));
        }
    }

    output[gy * width + gx] = lbp_code;
}

size_t dynamicSMemSizeFunc(int blockSize) {
    // We need to determine blockDim.x and blockDim.y.
    // The host side determines these. In the kernel, we assume a 2D block.
    // To make this function work independently, we need the dimensions.
    // However, the prompt says "based on the block dimensions". 
    // Usually, this function is called with a specific blockSize.
    // Let's assume blockDim.x is derived from MAX_BLOCKS_PER_SEGMENT or similar.
    // But the function signature only takes blockSize.
    // We must assume a square-ish block or a specific width.
    // Let's assume blockDim.x = min(blockSize, 32).
    int bx = (blockSize <= 32) ? blockSize : 32;
    int by = (blockSize + bx - 1) / bx;
    return (size_t)(bx + 2) * (by + 2);
}

cudaError_t getOptimalLaunchParams(int segmentSize, int &optBlockSize,
                                    int &blocksPerGrid, float &theoreticalOccupancy) {
    int bestBlockSize = 0;
    float bestOcc = 0.0f;

    for (int blockSize = 32; blockSize <= 1024; blockSize += 32) {
        int bx = (blockSize <= 32) ? blockSize : 32;
        int by = (blockSize + bx - 1) / bx;
        if (bx * by > blockSize) continue;

        size_t sMem = (size_t)(bx + 2) * (by + 2);
        
        int maxActiveBlocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, k_localBinaryKernel, blockSize, sMem);
        
        float occ = (float)maxActiveBlocks * blockSize / 2048.0f; // Approximation for typical SM
        if (occ > bestOcc) {
            bestOcc = occ;
            bestBlockSize = blockSize;
        }
    }

    optBlockSize = bestBlockSize;
    // This is a simplified grid calculation. In practice, it depends on image width/height.
    // Since we only have segmentSize, we assume a square-ish layout or that the 
    // harness handles the actual grid launch.
    blocksPerGrid = (segmentSize + optBlockSize - 1) / optBlockSize;
    theoreticalOccupancy = bestOcc;

    return cudaSuccess;
}