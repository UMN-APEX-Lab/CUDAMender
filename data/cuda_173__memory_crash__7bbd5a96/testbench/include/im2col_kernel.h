#ifndef IM2COL_KERNEL_H
#define IM2COL_KERNEL_H

#include <cuda_bf16.h>

// CUDA kernel to perform im2Col transformation
__global__ void k_im2ColTransform(__nv_bfloat16 *input,
                                  __nv_bfloat16 *weights,
                                  __nv_bfloat16 *im2ColPad,
                                  __nv_bfloat16 *weightPad,
                                  int channels,
                                  int inputHeight,
                                  int inputWidth,
                                  int kernelHeight,
                                  int kernelWidth,
                                  int pad,
                                  int stride,
                                  int outHeight,
                                  int outWidth,
                                  int numFilters,
                                  int padK);

#endif // IM2COL_KERNEL_H