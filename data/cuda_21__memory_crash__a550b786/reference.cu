// file: solution.cu
#include "stencil3d.h"

__global__ void stencil3d_kernel(float *input, float *output, unsigned int N)
{
    // istart is the starting index in the z-axis (depth), j,k are the row and column indices for
    // the current thread
    int istart = blockDim.z * OUT_TILE_DIM;
    int j      = blockDim.y * blockIdx.y + threadIdx.y - 1;
    int k      = blockDim.x * blockIdx.x + threadIdx.x - 1;

    // shared memory for the input tiles
    __shared__ float inprev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float incurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float innext_s[IN_TILE_DIM][IN_TILE_DIM];

    // Load the previous and the current tiles into shared memory if they are within the boundary
    if (istart - 1 >= 0 && istart - 1 < N && j >= 0 && j < N && k >= 0 & k < N)
    {
        inprev_s[threadIdx.y][threadIdx.x] = input[(istart - 1) * N * N + j * N + k];
    }
    if (istart >= 0 && istart < N && j >= 0 && j < N && k >= 0 && k < N)
    {
        incurr_s[threadIdx.y][threadIdx.x] = input[istart * N * N + j * N + k];
    }

    // Make sure all the data is available
    __syncthreads();

    // Perform the stencil computation for each of the output tile.
    for (int i = istart; i < istart + OUT_TILE_DIM; i++)
    {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N)
        {
            innext_s[threadIdx.y][threadIdx.x] = input[(istart + 1) * N * N + j * N + k];
        }

        __syncthreads();

        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1)
        {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 &&
                threadIdx.x < IN_TILE_DIM - 1)
            {
                output[i * N * N + j * N + k] = C0 * incurr_s[threadIdx.y][threadIdx.x];
                output[i * N * N + j * N + k] += C1 * (incurr_s[threadIdx.y][threadIdx.x - 1] +
                                                       incurr_s[threadIdx.y][threadIdx.x + 1] +
                                                       incurr_s[threadIdx.y - 1][threadIdx.x] +
                                                       incurr_s[threadIdx.y + 1][threadIdx.x] +
                                                       inprev_s[threadIdx.y][threadIdx.x] +
                                                       innext_s[threadIdx.y][threadIdx.x]);
            }
        }

        // Update the shared memory tiles for the next tile along the z-axis.
        __syncthreads();
        inprev_s[threadIdx.y][threadIdx.x] = incurr_s[threadIdx.y][threadIdx.x];
        incurr_s[threadIdx.y][threadIdx.x] = innext_s[threadIdx.y][threadIdx.x];
    }
}
