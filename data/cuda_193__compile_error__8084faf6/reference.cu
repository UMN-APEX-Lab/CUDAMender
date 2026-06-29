#include "mma_kernel.h"
#include <cstdint>

__global__ void k_mmaM16N8K16ArowBcol(__nv_bfloat16 *rowMajorA_d,
                                     __nv_bfloat16 *colMajorB_d,
                                     float          *resultMatrixC_d,
                                     int             mDim,
                                     int             nDim,
                                     int             kDim) {

    // Compute warp and lane indices
    extern __shared__ __nv_bfloat16 shmem[];       // WARPS_PER_BLOCK * (MMA_M*MMA_K + MMA_K*MMA_N)
    int warpsPerBlock  = blockDim.x / warpSize;
    int warpId         = threadIdx.x / warpSize;   // which warp within this block
    int lane           = threadIdx.x % warpSize;   // lane within the warp
    int lanesPerSubRow = 4;

    // Tile counts
    int numTilesM = (mDim + MMA_M - 1) / MMA_M;
    int numTilesN = (nDim + MMA_N - 1) / MMA_N;

    // Compute global tile indices for this warp
    int tileM = blockIdx.y * warpsPerBlock + warpId;  // row‐tile index
    int tileN = blockIdx.x;                           // col‐tile index

    // Guard
    if (tileM >= numTilesM)
                return;
    if (tileN >= numTilesN)
                return;

    // Each warp needs (MMA_M*MMA_K + MMA_K*MMA_N) __nv_bfloat16's
    int aTileSize = MMA_M * MMA_K;  // 16×8
    int bTileSize = MMA_K * MMA_N;  // 16×8

    __nv_bfloat16 *aTile = shmem + warpId * (aTileSize + bTileSize);
    __nv_bfloat16 *bTile = aTile + aTileSize;

    // Compute row/col offset in the full matrix
    int aRow = tileM * MMA_M;  // starting row of A for this warp
    int bCol = tileN * MMA_N;  // starting column of B for this warp

    int numTilesK = (kDim + MMA_K - 1) / MMA_K;

    // Fragment accumulator (4 f32 registers per warp)
    float accReg[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Loop over K‐tiles
    for (int tk = 0; tk < numTilesK; ++tk) {
        // Base pointers into global memory (row‐major A, col‐major B)
        // A (size: mDim × kDim), stored row‐major
        __nv_bfloat16 *aPtr = rowMajorA_d + aRow * kDim + tk * MMA_K;
        // B (size: kDim × nDim), stored col‐major → element (r,c) is at B[c*kDim + r]
        __nv_bfloat16 *bPtr = colMajorB_d + bCol   * kDim + tk * MMA_K * 1;

        // Warp‐strided load of A‐tile into shared memory
        // Warp‐stride (i += warpSize) load over aTileSize elements
        for (int i = lane; i < aTileSize; i += warpSize) {
            int r = i / MMA_K;   // 0..15
            int c = i % MMA_K;   // 0..7
            int globalRow = aRow + r;
            int globalCol = tk * MMA_K + c;  // current K‐tile column
            if (globalRow < mDim && globalCol < kDim) {
                aTile[i] = aPtr[r * kDim + c];
            } else {
                aTile[i] = __nv_bfloat16(0);
            }
        }

        // Warp‐strided load of B‐tile into shared memory
        for (int i = lane; i < bTileSize; i += warpSize) {
            int r = i / MMA_N;   // 0..7
            int c = i % MMA_N;   // 0..7
            int globalRow = tk * MMA_K + r;  // current K‐tile row
            int globalCol = bCol + c;        // current N‐tile column
            if (globalRow < kDim && globalCol < nDim) {
                bTile[i] = bPtr[c * kDim + r];
            } else {
                bTile[i] = __nv_bfloat16(0);
            }
        }


        __syncthreads();

        unsigned long long address;
        asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(address) : "l"(aTile));
        uint32_t aTileByteOffset = static_cast<uint32_t>(address)
                                + (lane % MMA_M) * MMA_K * sizeof(__nv_bfloat16);
        unsigned long long aTileAddr64 = static_cast<unsigned long long>(aTileByteOffset);

        uint32_t a0, a1, a2, a3;
        // first half (cols 0‐7)
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(a0), "=r"(a1) : "l"(aTileAddr64)
        );

        // second half (cols 8‐15)
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(a2), "=r"(a3)
            : "l"(aTileAddr64 + 8 * sizeof(__nv_bfloat16))
        );

        asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(address) : "l"(bTile));
        uint32_t bTileByteOffset = static_cast<uint32_t>(address)
                                + (lane % MMA_K) * MMA_N * sizeof(__nv_bfloat16);
        unsigned long long bTileAddr64 = static_cast<unsigned long long>(bTileByteOffset);

        uint32_t b0, b1;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
            : "=r"(b0), "=r"(b1) : "l"(bTileAddr64)
        );

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
            : "+f"(accReg[0]), "+f"(accReg[1]), "+f"(accReg[2]), "+f"(accReg[3])
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
            "r"(b0),"r"(b1)
        );


    } // end for each K‐tile

    float *cTile = resultMatrixC_d + (aRow * nDim) + bCol;
    int r = lane / lanesPerSubRow;              // 0..7
    int c = (lane % lanesPerSubRow) * 2;        // 0,2,4,6
    cTile[r * nDim + c] = accReg[0];
    cTile[r * nDim + c + 1] = accReg[1];
    cTile[(r + 8) * nDim + c] = accReg[2];
    cTile[(r + 8) * nDim + c + 1] = accReg[3];
}