#include "image_erosion.h"
#include "erosion_constants.h"

__device__ volatile int *globalBarrier = (volatile int *)0;

__global__ void k_imageErosion(const unsigned char *inputImage_d,
                               const unsigned char *structuringElement_d,
                               unsigned char *outputImage_d,
                               unsigned char *outputImageIntermediateBuffer_d,
                               int inputImageWidth,
                               int inputImageHeight,
                               int structuringElementWidth,
                               int structuringElementHeight,
                               int numberOfErosionIterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int totalThreads = gridDim.x * blockDim.x * gridDim.y * blockDim.y;
    int threadId = blockIdx.y * blockDim.y * gridDim.x * blockDim.x + 
                   blockIdx.x * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    
    int seCenterX = structuringElementWidth / 2;
    int seCenterY = structuringElementHeight / 2;
    
    unsigned char *readBuffer;
    unsigned char *writeBuffer;
    
    for (int iter = 0; iter < numberOfErosionIterations; iter++) {
        if (iter == 0) {
            readBuffer = (unsigned char*)inputImage_d;
        } else {
            readBuffer = outputImageIntermediateBuffer_d;
        }
        
        if (iter % 2 == 0) {
            writeBuffer = outputImageIntermediateBuffer_d;
        } else {
            writeBuffer = outputImage_d;
        }
        
        if (x < inputImageWidth && y < inputImageHeight) {
            int outputValue = 1;
            
            for (int seY = 0; seY < structuringElementHeight; seY++) {
                for (int seX = 0; seX < structuringElementWidth; seX++) {
                    int seIndex = seY * structuringElementWidth + seX;
                    if (structuringElement_d[seIndex] == 1) {
                        int imgX = x + seX - seCenterX;
                        int imgY = y + seY - seCenterY;
                        
                        if (imgX >= 0 && imgX < inputImageWidth && 
                            imgY >= 0 && imgY < inputImageHeight) {
                            int imgIndex = imgY * inputImageWidth + imgX;
                            if (readBuffer[imgIndex] != 1) {
                                outputValue = 0;
                                break;
                            }
                        }
                    }
                }
                if (outputValue == 0) break;
            }
            
            int imgIndex = y * inputImageWidth + x;
            writeBuffer[imgIndex] = (unsigned char)outputValue;
        }
        
        __threadfence_system();
        
        if (threadId == 0) {
            atomicExch((int*)globalBarrier, 0);
        }
        __syncthreads();
        
        atomicAdd((int*)globalBarrier, 1);
        __threadfence_system();
        
        if (threadId != 0) {
            while (atomicAdd((int*)globalBarrier, 0) < totalThreads) {
                __threadfence_system();
            }
        }
        __syncthreads();
    }
    
    if (numberOfErosionIterations % 2 == 0 && numberOfErosionIterations > 0) {
        if (x < inputImageWidth && y < inputImageHeight) {
            int imgIndex = y * inputImageWidth + x;
            outputImage_d[imgIndex] = outputImageIntermediateBuffer_d[imgIndex];
        }
    }
}