#include "include/mma_kernel.h"
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __device__ __forceinline__ unsigned cvta_to_shared_u32(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

static __device__ __forceinline__ void ldmatrix_x4_trans(unsigned &r0, unsigned &r1, unsigned &r2, unsigned &r3, unsigned addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
#else
    r0 = r1 = r2 = r3 = 0;
#endif
}

static __device__ __forceinline__ void ldmatrix_x2_trans(unsigned &r0, unsigned &r1, unsigned addr) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr));
#else
    r0 = r1 = 0;
#endif
}

static __device__ __forceinline__ void mma_m16n8k16_bf16(
    float &d0, float &d1, float &d2, float &d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#else
    d0 = c0; d1 = c1; d2 = c2; d3 = c3;
#endif
}

__global__ void k_mmaM16N8K16ArowBcol(
    __nv_bfloat16 *rowMajorA_d,
    __nv_bfloat16 *colMajorB_d,
    float *resultMatrixC_d,
    int mDim, int nDim, int kDim)
{
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(16) __nv_bfloat16 sA[MMA_M * MMA_K];
    __shared__ __align__(16) __nv_bfloat16 sB[MMA_K * MMA_N];

    const int lane = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    const int tile_m = blockIdx.y * MMA_M;
    const int tile_n = blockIdx.x * MMA_N;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    for (int k0 = 0; k0 < kDim; k0 += MMA_K) {
        for (int idx = lane; idx < MMA_M * MMA_K; idx += 32) {
            int r = idx / MMA_K;
            int c = idx % MMA_K;
            int gm = tile_m + r;
            int gk = k0 + c;
            sA[idx] = (gm < mDim && gk < kDim) ? rowMajorA_d[gm * kDim + gk] : __float2bfloat16(0.0f);
        }

        for (int idx = lane; idx < MMA_K * MMA_N; idx += 32) {
            int k = idx / MMA_N;
            int n = idx % MMA_N;
            int gk = k0 + k;
            int gn = tile_n + n;
            sB[idx] = (gk < kDim && gn < nDim) ? colMajorB_d[gk + gn * kDim] : __float2bfloat16(0.0f);
        }

        __syncthreads();

        unsigned a0, a1, a2, a3;
        unsigned b0, b1;

        unsigned addrA;
        if (lane < 8) {
            addrA = cvta_to_shared_u32(&sA[(lane) * MMA_K + 0]);
        } else if (lane < 16) {
            addrA = cvta_to_shared_u32(&sA[(lane - 8) * MMA_K + 8]);
        } else if (lane < 24) {
            addrA = cvta_to_shared_u32(&sA[(lane - 16 + 8) * MMA_K + 0]);
        } else {
            addrA = cvta_to_shared_u32(&sA[(lane - 24 + 8) * MMA_K + 8]);
        }

        unsigned addrB;
        if (lane < 8) {
            addrB = cvta_to_shared_u32(&sB[(lane) * MMA_N + 0]);
        } else if (lane < 16) {
            addrB = cvta_to_shared_u32(&sB[(lane - 8 + 8) * MMA_N + 0]);
        } else {
            addrB = cvta_to_shared_u32(&sB[0]);
        }

        ldmatrix_x4_trans(a0, a1, a2, a3, addrA);
        ldmatrix_x2_trans(b0, b1, addrB);

        float d0, d1, d2, d3;
        mma_m16n8k16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
        c0 = d0; c1 = d1; c2 = d2; c3 = d3;

        __syncthreads();
    }

    const int group = lane >> 2;
    const int tid4 = lane & 3;

    const int row0 = tile_m + group;
    const int row1 = tile_m + group + 8;
    const int col0 = tile_n + tid4 * 2;
    const int col1 = col0 + 1;

    if (row0 < mDim && col0 < nDim) resultMatrixC_d[row0 * nDim + col0] = c0;
    if (row0 < mDim && col1 < nDim) resultMatrixC_d[row0 * nDim + col1] = c1;
    if (row1 < mDim && col0 < nDim) resultMatrixC_d[row1 * nDim + col0] = c2;
    if (row1 < mDim && col1 < nDim) resultMatrixC_d[row1 * nDim + col1] = c3;
#else
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < mDim && col < nDim) {
        float sum = 0.0f;
        for (int k = 0; k < kDim; ++k) {
            sum += __bfloat162float(rowMajorA_d[row * kDim + k]) *
                   __bfloat162float(colMajorB_d[k + col * kDim]);
        }
        resultMatrixC_d[row * nDim + col] = sum;
    }
#endif
}