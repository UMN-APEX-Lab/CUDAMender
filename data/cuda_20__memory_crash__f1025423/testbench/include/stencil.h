#ifndef STENCIL_H
#define STENCIL_H

const int BLOCK_SIZE = 256;
const int RADIUS     = 3;

__global__ void stencil_1d(int *in, int *out);

#endif // STENCIL_H