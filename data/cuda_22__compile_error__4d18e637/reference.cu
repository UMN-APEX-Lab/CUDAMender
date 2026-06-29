#include "transpose.h"

__global__ void transpose(const float *input, float *output, int width, int height)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (x < width && y < height)
    {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    __syncthreads();

    x = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    y = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    if (x < height && y < width)
    {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}