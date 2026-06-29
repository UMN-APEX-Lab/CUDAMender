#ifndef KERNEL_H
#define KERNEL_H

#include <cstdint>

__global__ void k_mmaTensorMatMulM16N8k16Int8(uint8_t *inputMatrixA_d, 
                                              uint8_t *inputMatrixB_d, 
                                              int *outputMatrix_d, 
                                              int mDim, 
                                              int nDim, 
                                              int kDim);

#endif // KERNEL_H