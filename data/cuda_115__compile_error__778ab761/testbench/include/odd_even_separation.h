#ifndef ODD_EVEN_SEPARATION_H
#define ODD_EVEN_SEPARATION_H

#include <cuda_runtime.h>

constexpr int NUM_THREADS_PER_BLOCK = 256;

__global__ void k_separateOddEven(int *input_d, int *oddData_d, int *evenData_d, int numElements);

#endif // ODD_EVEN_SEPARATION_H