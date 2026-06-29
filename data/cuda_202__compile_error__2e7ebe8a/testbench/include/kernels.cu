#include "graph_pipeline.h"

// RGB to grayscale conversion
__global__ void k_rgbToGray(float* inpFrameRGB, int frameWidth, int frameHeight, float* outFrame) {
    // Thread/Block boundary checks
    for (int tIdy = (blockIdx.y * blockDim.y) + threadIdx.y; tIdy < frameHeight; tIdy += gridDim.y * blockDim.y) {
        for (int tIdx = (blockIdx.x * blockDim.x) + threadIdx.x; tIdx < frameWidth; tIdx += gridDim.x * blockDim.x) {
            // RGB to grayscale
            float grayPixValue = inpFrameRGB[tIdy * frameWidth + tIdx] * RGB2GRAY_R +
                inpFrameRGB[tIdy * frameWidth + tIdx + frameWidth * frameHeight] * RGB2GRAY_G +
                inpFrameRGB[tIdy * frameWidth + tIdx + 2 * frameWidth * frameHeight] * RGB2GRAY_B;
            outFrame[tIdy * frameWidth + tIdx] = max(0.f, min(1.f, grayPixValue));
        }
    }
}

// Gamma Correction, contrast and brightness adjustment
__global__ void k_gammaCorrection(float* inpFrame, int frameWidth, int frameHeight, float gammaValue, float contrastValue, float brightnessValue, float* outFrame) {
    // Thread/Block boundary checks
    for (int tIdy = (blockIdx.y * blockDim.y) + threadIdx.y; tIdy < frameHeight; tIdy += gridDim.y * blockDim.y) {
        for (int tIdx = (blockIdx.x * blockDim.x) + threadIdx.x; tIdx < frameWidth; tIdx += gridDim.x * blockDim.x) {
            // Gamma Correction
            float pixValueAfterGamma = max(0.f, min(1.f, pow(inpFrame[tIdy * frameWidth + tIdx], gammaValue)));
            pixValueAfterGamma = contrastValue * pixValueAfterGamma + brightnessValue;
            outFrame[tIdy * frameWidth + tIdx] = min(max(pixValueAfterGamma, 0.f), 1.f);
        }
    }
}