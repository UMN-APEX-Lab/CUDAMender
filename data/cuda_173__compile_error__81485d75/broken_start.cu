#include <cuda_bf16.h>
#include <mma.h>
#include "mma_tensor_conv.h"

using namespace nvcuda;

__global__ void k_mmaTensorConvMatMul(__nv_bfloat16 *inputMatrixA_d, 
                                      __nv_bfloat16 *inputMatrixB_d, 
                                      float *outputMatrix_d, 
                                      int mDim, int nDim, int kDim)
{
    // Define MMA shape
    constexpr int M = 16;
    constexpr int N = 8;
    constexpr int K = 8;
    
    // Thread block configuration
    constexpr int threadsPerBlock = 128;
    constexpr int warpsPerBlock = threadsPerBlock / 32;
    
    // Each warp handles one 16x8 output tile
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    // Calculate output tile position
    int rowOffset = blockIdx.y * M + (warpId / (nDim / N)) * M;
    int colOffset = blockIdx.x * N + (warpId % (nDim / N)) * N;
    
    // Shared memory for A and B matrices
    extern __shared__ __nv_bfloat16 sharedMem[];
    __nv_bfloat16 *sharedA = sharedMem;
    __nv_bfloat16 *sharedB = sharedMem + (M * K * warpsPerBlock);
    
    // Register arrays for A and B tiles
    __nv_bfloat16 regA[4];  // 16x8 = 128 bf16 elements = 4x32 bf16
    __nv_bfloat16 regB[2];  // 8x8 = 64 bf16 elements = 2x32 bf16
    float regC[4];          // 16x8 f32 output tile
    
    // Initialize output register
    for (int i = 0; i < 4; i++) {
        regC[i] = 0.0f;
    }
    
    // Convert shared memory pointers to u32 for ldmatrix
    uint32_t sharedA_ptr = d_cvtaToSharedU32(sharedA);
    uint32_t sharedB_ptr = d_cvtaToSharedU32(sharedB);
    
    // Calculate tile index for this warp
    int warpTileRow = (warpId / (nDim / N));
    int warpTileCol = (warpId % (nDim / N));
    
    // Load data from global memory to shared memory
    for (int k = 0; k < kDim; k += K) {
        // Load A tile (M x K)
        for (int i = 0; i < M; i++) {
            int row = warpTileRow * M + i;
            if (row < mDim) {
                int col = k;
                int idx = row * kDim + col;
                sharedA[(warpTileRow * K) * M + i * K + (k % K)] = 
                    (k < kDim) ? inputMatrixA_d[idx] : __nv_bfloat16(0.0f);
            }
        }
        
        // Load B tile (K x N)
        for (int j = 0; j < N; j++) {
            int col = warpTileCol * N + j;
            if (col < nDim) {
                int row = k;
                int idx = row * nDim + col;
                sharedB[(k % K) * N + j] = 
                    (k < kDim) ? inputMatrixB_d[idx] : __nv_bfloat16(0.0f);
            }
        }
        
        __syncthreads();
        
        // Load A from shared memory to registers using ldmatrix
        uint32_t ldA_ptr = sharedA_ptr + (warpTileRow * M * K + (k % K) * M) * 2;
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(regA[0]), "=r"(regA[1]), "=r"(regA[2]), "=r"(regA[3])
            : "r"(ldA_ptr)
        );
        
        // Load B from shared memory to registers using ldmatrix
        uint32_t ldB_ptr = sharedB_ptr + (k % K) * N * 2;
        asm volatile(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"
            : "=r"(regB[0]), "=r"(regB[1])
            : "r"(ldB_ptr)
        );
        
        // Perform MMA operation: C += A * B
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
            : "=f"(regC[0]), "=f"(regC[1]), "=f"(regC[2]), "=f"(regC[3])
            : "r"(regA[0]), "r"(regA[1]), "r"(regA[2]), "r"(regA[3]),
              "r"(regB[0]), "r"(regB[1]),
              "f"(regC[0]), "f"(regC[1]), "f"(regC[2]), "f"(regC[3])
        );
        
        __syncthreads();
    }
    
    // Store the result to global memory
    int outRow = warpTileRow * M;
    int outCol = warpTileCol * N;
    
    if (outRow < mDim && outCol < nDim) {
        d_storeMatrixTile16x8_f32(&outputMatrix_d[outRow * nDim + outCol], regC, nDim);
    }
}