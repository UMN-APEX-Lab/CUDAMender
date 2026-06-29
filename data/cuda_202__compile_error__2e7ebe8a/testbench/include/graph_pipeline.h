#ifndef GRAPH_PIPELINE_H
#define GRAPH_PIPELINE_H

#include <iostream>
#include <vector>
#include <time.h>
#include <assert.h>
#include <cuda_runtime_api.h>

#undef NDEBUG
#define _USE_MATH_DEFINES

#define RGB_CHANNELS 3
#define RGB2GRAY_R 0.2989
#define RGB2GRAY_G 0.5870
#define RGB2GRAY_B 0.1140

#define CUDA_CHECK(call)                                    \
do {                                                        \
    cudaError_t error = call;                               \
    if (error != cudaSuccess) {                             \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",       \
            __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(0)

// Main function to implement
void runGraph(float* inpFrame, float* inpFrame_d, size_t frameHeight, size_t frameWidth, 
              size_t numFrames, float gammaValue, float contrastValue, float brightnessValue, 
              float* outRGB2Gray_d, float* outFrame_d, float* outFrame_h, cudaStream_t& stream);

// Helper kernels (implementations in kernels.cu)
__global__ void k_rgbToGray(float* inpFrameRGB, int frameWidth, int frameHeight, float* outFrame);
__global__ void k_gammaCorrection(float* inpFrame, int frameWidth, int frameHeight, 
                                   float gammaValue, float contrastValue, float brightnessValue, 
                                   float* outFrame);

#endif // GRAPH_PIPELINE_H