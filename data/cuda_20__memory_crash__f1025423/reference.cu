// file: solution.cu
#include "stencil.h"

// This CUDA kernel performs a 1D stencil operation on a large 1D array.
// The stencil operation calculates the sum of each element and its neighboring elements within a
// specified radius. The function handles array sizes larger than the number of threads in a block
// and utilizes shared memory for optimization.
__global__ void stencil_1d(int *in, int *out)
{
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS)
    {
        temp[lindex - RADIUS] = (gindex - RADIUS >= 0) ? in[gindex - RADIUS] : 0;
        temp[lindex + BLOCK_SIZE] =
            (gindex + BLOCK_SIZE < gridDim.x * blockDim.x) ? in[gindex + BLOCK_SIZE] : 0;
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)
    {
        result += temp[lindex + offset];
    }

    // Store the result
    out[gindex] = result;
}
