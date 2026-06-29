#ifndef K_JOIN_IMAGES_H
#define K_JOIN_IMAGES_H

#include <cuda_runtime.h>

typedef unsigned char uchar;

__global__ void k_joinImages(const uchar* image1_d, 
                             const uchar* image2_d, 
                             uchar*       outputImage_d, 
                             int          width,
                             int          height);

#endif // K_JOIN_IMAGES_H