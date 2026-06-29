#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "include/mma_kernel.h"

__global__ void k_mmaM16N8K8AcolBrow(
    __nv_bfloat16 *colMajorA_d,
    __nv_bfloat16 *rowMajorB_d,
    float *resultMatrixC_d,
    int mDim,
    int nDim,
    int kDim)
{
    // MMA m16n8k8: m=16, n=8, k=8
    // Each warp handles 16x8 tile of C
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpM = (warpId * 8) % mDim; // Example tiling: adjust based on actual block layout
    // For simplicity in this specific task context, we assume 1 warp per 16x8 block 
    // or a simple mapping. Let's use a standard 2D mapping.
    
    int warp_m = (blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32)) * 16;
    int warp_n = (blockIdx.y * (blockDim.y / 32) + (threadIdx.x % 32)) * 8; // This is a placeholder logic
    
    // Correct warp mapping for a single block
    int tid = threadIdx.x;
    int laneId = tid % 32;
    int warpId_local = tid / 32;
    
    // We assume the grid/block is configured such that each warp handles a 16x8 tile
    // Let's use the blockIdx for global tile positioning
    int m_start = blockIdx.x * 16;
    int n_start = blockIdx.y * 8;

    __shared__ __nv_bfloat16 sA[16][8]; // Tile for A (m=16, k=8)
    __shared__ __nv_bfloat16 sB[8][8];  // Tile for B (k=8, n=8)

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_offset = 0; k_offset < kDim; k_offset += 8) {
        // Load A (Column Major): A[mDim][kDim]
        // A is Col Major: A[i][j] is at colMajorA_d[j * mDim + i]
        // We need a 16x8 tile. 
        // Thread mapping for loading shared memory: 32 threads to load 16*8 = 128 elements.
        // Each thread loads 4 elements.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = laneId + i * 32;
            if (load_idx < 128) {
                int row = load_idx % 16;
                int col = load_idx / 16;
                int global_row = m_start + row;
                int global_col = k_offset + col;
                if (global_row < mDim && global_col < kDim)
                    sA[row][col] = colMajorA_d[global_col * mDim + global_row];
                else
                    sA[row][col] = __float2bfloat16(0.0f);
            }
        }

        // Load B (Row Major): B[kDim][nDim]
        // B is Row Major: B[i][j] is at rowMajorB_d[i * nDim + j]
        // We need an 8x8 tile. 64 elements.
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int load_idx = laneId + i * 32;
            if (load_idx < 64) {
                int row = load_idx / 8;
                int col = load_idx % 8;
                int global_row = k_offset + row;
                int global_col = n_start + col;
                if (global_row < kDim && global_col < nDim)
                    sB[row][col] = rowMajorB_d[global_row * nDim + global_col];
                else
                    sB[row][col] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        __nv_bfloat16 regA[4];
        __nv_bfloat16 regB[2];

        // ldmatrix.x4 loads 8 elements into registers
        // For A (16x8), we need 2 ldmatrix.x4 calls to fill registers for mma.m16n8k8
        // However, mma.m16n8k8 expects specific register layouts.
        // Using inline PTX:
        uint32_t sA_addr = d_cvtaToSharedU32(&sA[0][0]);
        uint32_t sB_addr = d_cvtaToSharedU32(&sB[0][0]);

        asm volatile(
            "ldmatrix.x4.bfloat16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(regA[0]), "=r"(regA[1]), "=r"(regA[2]), "=r"(regA[3])
            : "r"(sA_addr)
        );
        
        // For B, we need to load 8 elements. 
        // mma.m16n8k8 requires B in a specific format.
        asm volatile(
            "ldmatrix.x2.bfloat16 {%0, %1}, [%4];\n"
            : "=r"(regB[0]), "=r"(regB[1])
            : "r"(sB_addr)
        );

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, %10;\n"
            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
            : "r"(regA[0]), "r"(regA[1]), "r"(regA[2]), "r"(regA[3]),
              "r"(regB[0]), "r"(regB[1]), "r"(0) // predicate
        );
        __syncthreads();
    }

    // Store result C
    // Each warp produced a 16x8 tile.
    // Result C is stored in row-major or as specified by d_storeMatrixTile16x8_f32.
    // The helper function handles the mapping of the 4 registers to the 16x8 output.
    d_storeMatrixTile16x8_f32(&resultMatrixC_d[m_start * nDim + n_start], acc, nDim);
}