#include "mma_kernel.h"
#include <mma.h>

__global__ void k_mmaM16N8K8AcolBrow(__nv_bfloat16 *colMajorA_d, __nv_bfloat16 *rowMajorB_d, float *resultMatrixC_d, int mDim, int nDim, int kDim) {
    // Tile indices.
    int blockM = blockIdx.y;
    int blockN = blockIdx.x;

    int aRow = blockM * MMA_M;  // starting row in colMajorA_d
    int bCol = blockN * MMA_N;  // starting column in rowMajorB_d

    // Allocate shared memory for one tile of colMajorA_d (16×8) and one tile of rowMajorB_d (8×8).
    extern __shared__ __nv_bfloat16 shmem[];
    __nv_bfloat16* aTile = shmem;                    // MMA_M * MMA_K elements
    __nv_bfloat16* bTile = shmem + MMA_M * MMA_K;    // MMA_K * MMA_N elements

    // Each thread holds a fragment of the output: 4 f32 values.
    float accReg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int numTilesK = kDim / MMA_K;

    for (int tileK = 0; tileK < numTilesK; tileK++) {
        // Global pointers for the current tile.
        __nv_bfloat16* aG = colMajorA_d + (tileK*MMA_K) * mDim + aRow;
        __nv_bfloat16* bG = rowMajorB_d + (tileK*MMA_K) * nDim + bCol;

        // Load colMajorA_d tile (16×8) into shared memory with bounds checking.
        for (int i = threadIdx.x; i < MMA_M * MMA_K; i += blockDim.x) {
            int r = i / MMA_K;
            int c = i % MMA_K;
            int global_r = aRow + r;
            int global_c = tileK * MMA_K + c;
            aTile[i] = (global_r < mDim && global_c < kDim) ? aG[c * mDim + r] : __nv_bfloat16(0.0f);
        }

        // Load rowMajorB_d tile (8×8) into shared memory with bounds checking.
        for (int i = threadIdx.x; i < MMA_K * MMA_N; i += blockDim.x) {
            int r = i / MMA_N;
            int c = i % MMA_N;
            int global_r = tileK * MMA_K + r;
            int global_c = bCol + c;
            bTile[i] = (global_r < kDim && global_c < nDim) ? bG[r * nDim + c] : __nv_bfloat16(0.0f);
        }
        __syncthreads();

        // Load fragments from shared memory using ldmatrix.
        uint32_t aRegTile[2];
        int lane = threadIdx.x % 32;
        int aTileByteOffset = d_cvtaToSharedU32(aTile) + ((lane % MMA_M) * MMA_K * sizeof(__nv_bfloat16));
        unsigned long long aTileByteOffset64 = static_cast<unsigned long long>(aTileByteOffset);
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                     : "=r"(aRegTile[0]), "=r"(aRegTile[1])
                     : "l"(aTileByteOffset64));

        uint32_t bRegTile;
        int bTileByteOffset = d_cvtaToSharedU32(bTile) + ((lane % MMA_K) * MMA_N * sizeof(__nv_bfloat16));
        unsigned long long bTileByteOffset64 = static_cast<unsigned long long>(bTileByteOffset);
        asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                     : "=r"(bRegTile)
                     : "l"(bTileByteOffset64));

        // Perform MMA: multiply the bf16 fragments and accumulate into f32 registers.
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                     : "+f"(accReg[0]), "+f"(accReg[1]), "+f"(accReg[2]), "+f"(accReg[3])
                     : "r"(aRegTile[0]), "r"(aRegTile[1]), "r"(bRegTile));
        __syncthreads();
    }

    // Write the computed output tile (16×8) to global memory.
    float* cTile = resultMatrixC_d + aRow * nDim + bCol;
    d_storeMatrixTile16x8_f32(cTile, accReg, nDim);
    __syncthreads();
}