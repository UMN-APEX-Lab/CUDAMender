#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

__global__ void apply_edge_detection(float* img, float* result, int width, int height);
__global__ void normalize_image(float* img, int width, int height);
__global__ void apply_blur_filter(float* img, float* result, int width, int height);
__global__ void combine_filtered_results(float* edge_result, float* blur_result, float* combined_result, int width, int height);
__global__ void final_transformation(float* img, int width, int height);

#endif // KERNELS_H