#include "stencil.h"

__global__ void stencil_1d(int *in, int *out) {
    __shared__ int s_data[BLOCK_SIZE + 2 * RADIUS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load the main element into shared memory
    s_data[tid + RADIUS] = in[gid];

    // Load the left halo
    if (tid < RADIUS) {
        int left_idx = gid - RADIUS + tid;
        s_data[tid] = (left_idx >= 0) ? in[left_idx] : 0;
    }

    // Load the right halo
    if (tid >= BLOCK_SIZE - RADIUS) {
        int offset = tid - (BLOCK_SIZE - RADIUS);
        int last_main_idx = blockIdx.x * BLOCK_SIZE + (BLOCK_SIZE - 1);
        int right_idx = last_main_idx + 1 + offset;
        // Since N is not provided, we assume the array is padded or bounds are managed.
        s_data[BLOCK_SIZE + RADIUS + offset] = in[right_idx];
    }

    __syncthreads();

    int sum = 0;
    for (int i = -RADIUS; i <= RADIUS; ++i) {
        sum += s_data[tid + RADIUS + i];
    }

    out[gid] = sum;
}
