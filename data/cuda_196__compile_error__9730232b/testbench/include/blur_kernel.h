#ifndef BLUR_KERNEL_H
#define BLUR_KERNEL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/std/array>
#include <cuda/std/cmath>

#define CUDA_CHECK(call) {                                     \
    cudaError_t error = call;                                  \
    if (error != cudaSuccess) {                                \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

// Settings.
constexpr int MAX_PIXELS = 10000;
constexpr int BLUR_WIDTH = 3;
constexpr int BLUR_HEIGHT = 3;
constexpr int BLUR_CALCULATIONS_PER_PIXEL = BLUR_WIDTH * BLUR_HEIGHT;
constexpr float COEFFICIENT_0 = 0.25f;
constexpr float COEFFICIENT_1 = 0.50f;
constexpr float COEFFICIENT_2 = 0.25f;
constexpr float COEFFICIENT_3 = 0.50f;
constexpr float COEFFICIENT_4 = 1.0f;
constexpr float COEFFICIENT_5 = 0.50f;
constexpr float COEFFICIENT_6 = 0.25f;
constexpr float COEFFICIENT_7 = 0.50f;
constexpr float COEFFICIENT_8 = 0.25f;

// Kernel declaration
__global__ void k_blur(int width, int height, cuda::std::array<float, MAX_PIXELS>* pixel_d, cuda::std::array<float, MAX_PIXELS>* blurredPixel_d);

#endif // BLUR_KERNEL_H