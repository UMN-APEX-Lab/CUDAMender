#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

__global__ void k_mmaM16N8K16AcolBcol(__nv_bfloat16 *colMajorA_d, 
                                      __nv_bfloat16 *colMajorB_d, 
                                      float *rowMajorC_d, 
                                      int mDim, int nDim, int kDim);

#endif // MATRIX_MUL_H