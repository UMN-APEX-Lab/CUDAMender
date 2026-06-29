#include "transpose.h"

__global__ void transpose(const float *input, float *output, int width, int height) {
    // Shared memory to hold a tile of the input matrix
    // BLOCK_SIZE is defined in transpose.h
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE];

    // Calculate the global row and column indices of the element to be read from input
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    // Read from input matrix and store in shared memory tile
    // Boundary check for non-square matrices larger than BLOCK_SIZE
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }

    // Synchronize to ensure the tile is fully populated
    __syncthreads();

    // Calculate the global row and column indices of the element to be written to output
    // The output matrix dimensions are (width x height) transposed to (height x width)
    // New coordinates: x_out = blockIdx.y * BLOCK_SIZE + threadIdx.x
    //                y_out = blockIdx.x * BLOCK_SIZE + threadIdx.y
    int x_out = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int y_out = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    // Boundary check for the output matrix
    if (x_out < height && y_out < width) {
        // Read from the shared memory tile in transposed order
        // The element at tile[threadIdx.y][threadIdx.x] was input[y][x]
        // We want output[x][y] = input[y][x]
        output[y_out * height + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}