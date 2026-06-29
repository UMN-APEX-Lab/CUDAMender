#include "include/k_joinImages.h"

__global__ void k_joinImages(const unsigned char* image1_d,
                             const unsigned char* image2_d,
                             unsigned char* outputImage_d,
                             int width,
                             int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    int outWidth = width * 2;

    for (int row = y; row < height; row += strideY) {
        int inRowOffset = row * width;
        int outRowOffset = row * outWidth;

        for (int col = x; col < width; col += strideX) {
            unsigned char v1 = image1_d[inRowOffset + col];
            unsigned char v2 = image2_d[inRowOffset + col];

            outputImage_d[outRowOffset + col] = v1;
            outputImage_d[outRowOffset + width + col] = v2;
        }
    }
}