#include "blur_kernel.h"
#include <cuda/std/array>

__global__ void k_blur(int width, int height,
                       cuda::std::array<float, MAX_PIXELS>* pixel_d,
                       cuda::std::array<float, MAX_PIXELS>* blurredPixel_d) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) {
        return;
    }
    
    int idx = y * width + x;
    
    // 3x3 filter weights
    // 0.25  0.50  0.25
    // 0.50  1.00  0.50
    // 0.25  0.50  0.25
    // Total weight = 4.0
    
    float sum = 0.0f;
    float total_weight = 0.0f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            
            float weight;
            if (dx == 0 && dy == 0) {
                weight = 1.0f;
            } else if (dx == 0 || dy == 0) {
                weight = 0.5f;
            } else {
                weight = 0.25f;
            }
            
            float pixel_value = 0.0f;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                pixel_value = (*pixel_d)[ny * width + nx];
            }
            
            sum += weight * pixel_value;
            total_weight += weight;
        }
    }
    
    (*blurredPixel_d)[idx] = sum / total_weight;
}