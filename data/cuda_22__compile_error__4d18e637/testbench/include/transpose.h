#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#define BLOCK_SIZE 16

__global__ void transpose(const float *input, float *output, int width, int height);

#endif // TRANSPOSE_H