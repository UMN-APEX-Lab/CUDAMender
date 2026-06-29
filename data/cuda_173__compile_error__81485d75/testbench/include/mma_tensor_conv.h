#ifndef MMA_TENSOR_CONV_H
#define MMA_TENSOR_CONV_H

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 8

// PTX MMA instruction based Matrix Multiplication Kernel
__global__ void k_mmaTensorConvMatMul(__nv_bfloat16 *inputMatrixA_d,
                                      __nv_bfloat16 *inputMatrixB_d,
                                      float *outputMatrix_d,
                                      int mDim,
                                      int nDim,
                                      int kDim);

// Device function to convert generic address to shared memory state space
__device__ __forceinline__ uint32_t d_cvtaToSharedU32(const void* ptr) {
    unsigned long long address;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(address) : "l"(ptr));
    return static_cast<uint32_t>(address);
}

// Device function to store 16x8 output of mma instruction
__device__ __forceinline__ void d_storeMatrixTile16x8_f32(float* dst, float reg[4], int n) {
    int lane = threadIdx.x % 32;  // 0..31
    int r = lane / 4;        // 0..7 for the top half
    int c = (lane % 4) * 2;    // columns: 0,2,4,6
    dst[r * n + c] = reg[0];
    dst[r * n + c + 1] = reg[1];
    dst[(r + 8) * n + c] = reg[2];
    dst[(r + 8) * n + c + 1] = reg[3];
}

#endif // MMA_TENSOR_CONV_H