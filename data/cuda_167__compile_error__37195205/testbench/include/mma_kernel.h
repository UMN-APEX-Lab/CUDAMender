#ifndef MMA_KERNEL_H
#define MMA_KERNEL_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define MMA_M 16
#define MMA_N 8
#define MMA_K 8

// Convert generic pointer to shared memory state space
__device__ __forceinline__ uint32_t d_cvtaToSharedU32(const void* ptr) {
    unsigned long long address;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(address) : "l"(ptr));
    return static_cast<uint32_t>(address);
}

// Store 16x8 mma operation output tile
__device__ __forceinline__ void d_storeMatrixTile16x8_f32(float* dst, float reg[4], int n) {
    int lane = threadIdx.x % 32;  // 0..31
    int r = lane / 4;        // 0..7 for the top half
    int c = (lane % 4) * 2;    // columns: 0,2,4,6
    dst[r * n + c] = reg[0];
    dst[r * n + c + 1] = reg[1];
    dst[(r + 8) * n + c] = reg[2];
    dst[(r + 8) * n + c + 1] = reg[3];
}

// Kernel: Multiply bf16 matrices colMajorA_d (mDim×kDim) and rowMajorB_d (kDim×nDim) 
// to produce output resultMatrixC_d (mDim×nDim) in f32
__global__ void k_mmaM16N8K8AcolBrow(
    __nv_bfloat16 *colMajorA_d, 
    __nv_bfloat16 *rowMajorB_d, 
    float *resultMatrixC_d, 
    int mDim, 
    int nDim, 
    int kDim);

#endif // MMA_KERNEL_H