#include "bilinear_interpolation.h"

__global__ void k_bilinearInterpolation(float *inputMat, int inputWidth, int inputHeight, float *outputMat, int outputWidth, int outputHeight) {
    // Loading clock and thread Ids
    int local_tidX = threadIdx.x;
    int local_tidY = threadIdx.y;
    int thX = (blockIdx.x * blockDim.x) + local_tidX;
    int thY = (blockIdx.y * blockDim.y) + local_tidY;

    // Computing the width and height ratios
    float xRatio = (inputWidth - 1) / (float)(outputWidth - 1);
    float yRatio = (inputHeight - 1) / (float)(outputHeight - 1);

    int xOffset = (int)(blockIdx.x * blockDim.x * xRatio);
    int yOffset = (int)(blockIdx.y * blockDim.y * yRatio);

    if ((thX >= outputWidth) || (thY >= outputHeight)) {
        return;
    }

    // Using shared memory to store input as multiple threads re-use the same input location
    extern __shared__ float sharedMem[];
    int xLim = ceilf(BLOCK_SIZE * xRatio) + 2;
    int yLim = ceilf(BLOCK_SIZE * yRatio) + 2;

    if ((local_tidX < xLim) && (local_tidY < yLim)) {
        int gx = min(xOffset + local_tidX, inputWidth - 1);
        int gy = min(yOffset + local_tidY, inputHeight - 1);
        sharedMem[local_tidY * xLim + local_tidX] = inputMat[gy * inputWidth + gx];
    }

    __syncthreads();

    // Computing bilinear interpolation
    float dx = (thX * xRatio);
    float dy = (thY * yRatio);
    int dy_l = (int)dy - yOffset; int dy_h = ceilf(dy) - yOffset;
    int dx_l = (int)dx - xOffset; int dx_h = ceilf(dx) - xOffset;

    float dxDiff = (dx - floorf(dx));
    float dyDiff = (dy - floorf(dy));

    float tmpX1 = ((1 - dxDiff) * sharedMem[dy_l * xLim + dx_l]) + (dxDiff * sharedMem[dy_l * xLim + dx_h]);
    float tmpX2 = ((1 - dxDiff) * sharedMem[dy_h * xLim + dx_l]) + (dxDiff * sharedMem[dy_h * xLim + dx_h]);
    float outVal = (1 - dyDiff) * tmpX1 + dyDiff * tmpX2;

    // Clip the outputs to the interval [0.0,1.0]
    outVal = (outVal > 1.0) ? 1.0 : (outVal < 0.0) ? 0.0 : outVal;
    outputMat[thY * outputWidth + thX] = outVal;
}