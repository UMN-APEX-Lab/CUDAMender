#include "k_joinImages.h"

__global__ void k_joinImages(const uchar* image1_d, 
                             const uchar* image2_d, 
                             uchar*       outputImage_d, 
                             int          width, 
                             int          height) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx < width && ty < height) {
        // Calculate the linear index for the current pixel
        int index = ty * width + tx;

        // Write image1 data to the left side of the output
        outputImage_d[ty * (width * 2) + tx] = image1_d[index];

        // Write image2 data to the right side of the output
        outputImage_d[ty * (width * 2) + (tx + width)] = image2_d[index];
    }
}