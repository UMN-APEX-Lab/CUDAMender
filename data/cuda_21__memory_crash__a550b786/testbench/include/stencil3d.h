#ifndef STENCIL3D_H
#define STENCIL3D_H

#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM ((IN_TILE_DIM)-2)

#define C0 0.95
#define C1 0.05

__global__ void stencil3d_kernel(float *input, float *output, unsigned int N);

#endif // STENCIL3D_H