#ifndef IMAGE_EROSION_H
#define IMAGE_EROSION_H

__global__ void k_imageErosion(const unsigned char *inputImage_d, 
                               const unsigned char *structuringElement_d, 
                               unsigned char *outputImage_d, 
                               unsigned char *outputImageIntermediateBuffer_d, 
                               int inputImageWidth, 
                               int inputImageHeight, 
                               int structuringElementWidth, 
                               int structuringElementHeight, 
                               int numberOfErosionIterations);

#endif // IMAGE_EROSION_H