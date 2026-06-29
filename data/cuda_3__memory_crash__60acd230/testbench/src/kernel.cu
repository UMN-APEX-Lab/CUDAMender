#include "kernel_launch.h"

__global__ void kernel(int *output, const int *input)
{
    int id     = threadIdx.x + blockIdx.x * blockDim.x;
    output[id] = input[id];
}