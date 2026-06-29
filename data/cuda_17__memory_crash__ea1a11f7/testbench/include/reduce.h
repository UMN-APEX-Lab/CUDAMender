#ifndef REDUCE_H
#define REDUCE_H

#include <stddef.h>

const int BLOCK_SIZE = 256;

__global__ void reduce(float *gdata, float *out, size_t n);

#endif // REDUCE_H