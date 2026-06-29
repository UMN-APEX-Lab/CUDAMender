#ifndef KERNEL_WRAPPER_H
#define KERNEL_WRAPPER_H

#include <vector>

__global__ void kernel(float *pOut, const float *pIn);
void launch_with_input(int blockSize, const std::vector<float>& input, float expected, const char* test_name);

#endif