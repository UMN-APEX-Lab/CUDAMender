#ifndef MMA_KERNEL_H
#define MMA_KERNEL_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

__global__ void k_mmaM16N8K16ArowBcol(__nv_bfloat16 *rowMajorA_d, 
                                      __nv_bfloat16 *colMajorB_d, 
                                      float *resultMatrixC_d, 
                                      int mDim, 
                                      int nDim, 
                                      int kDim);

#endif // MMA_KERNEL_H