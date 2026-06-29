#include "blur_kernel.h"

__global__ void k_blur(int width, int height, cuda::std::array<float, MAX_PIXELS>* pixel_d, cuda::std::array<float, MAX_PIXELS>* blurredPixel_d) {
    // Blur coefficients.
    cuda::std::array<float, BLUR_CALCULATIONS_PER_PIXEL> blurCoefficients_d = { COEFFICIENT_0, COEFFICIENT_1, COEFFICIENT_2, COEFFICIENT_3, COEFFICIENT_4, COEFFICIENT_5, COEFFICIENT_6, COEFFICIENT_7, COEFFICIENT_8 };
    // Blur indices.
    cuda::std::array<int, BLUR_CALCULATIONS_PER_PIXEL> blurOffsetX_d = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
    cuda::std::array<int, BLUR_CALCULATIONS_PER_PIXEL> blurOffsetY_d = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
    float inverseOfSumOfCoefficients = 1.0f / (COEFFICIENT_0 + COEFFICIENT_1 + COEFFICIENT_2 + 
                                               COEFFICIENT_3 + COEFFICIENT_4 + COEFFICIENT_5 + 
                                               COEFFICIENT_6 + COEFFICIENT_7 + COEFFICIENT_8);
    int globalThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int numberOfGlobalThreads = blockDim.x * gridDim.x;
    int numberOfGridStrideLoopIterations = (width * height + numberOfGlobalThreads - 1) / numberOfGlobalThreads;
    for(int gridStrideIteration = 0; gridStrideIteration < numberOfGridStrideLoopIterations; gridStrideIteration++) {
        int pixelIndex = gridStrideIteration * numberOfGlobalThreads + globalThreadIndex;
        int pixelPositionX = pixelIndex % width;
        int pixelPositionY = pixelIndex / width;
        if (pixelPositionX < width && pixelPositionY < height) {
            int arrayIndex = 0;
            float color = 0.0f;
            // Applying a blur operation to the pixel at (pixelPositionX, pixelPositionY) by utilizing cuda::std::array for the offsets of neighboring pixels and the coefficients of the blur operation.
            for(auto coeff : blurCoefficients_d) { 
                int neighborX = pixelPositionX + blurOffsetX_d[arrayIndex];
                int neighborY = pixelPositionY + blurOffsetY_d[arrayIndex];
                if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                    color = cuda::std::fmaf(coeff, (*pixel_d)[neighborX + neighborY * width], color);
                }
                arrayIndex++;
            };
            (*blurredPixel_d)[pixelPositionX + pixelPositionY * width] = color * inverseOfSumOfCoefficients;
        }
    }
}