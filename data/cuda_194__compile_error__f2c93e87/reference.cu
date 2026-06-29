#include "matrix_mul.h"
#include <cstdint>

__global__ void k_mmaM16N8K16AcolBcol(__nv_bfloat16* colMajorA_d,      // column-major [mDim × kDim]
                                     __nv_bfloat16* colMajorB_d,      // [kDim × nDim]
                                     float* C,      // row-major [mDim × nDim]
                                     int mDim,
                                     int nDim,
                                     int kDim) {

    // warp‐splitting
    int warpId      = threadIdx.x / warpSize;
    int lane        = threadIdx.x % warpSize;
    int warpsPerBlock = blockDim.x / warpSize;
    int lanesPerSubRow = 4;

    // Tile counts
    int numTilesM = (mDim + MMA_M - 1) / MMA_M;
    int numTilesN = (nDim + MMA_N - 1) / MMA_N;
    int numTilesK   = (kDim + MMA_K - 1) / MMA_K;

    // Compute global tile indices for this warp
    int tileM = blockIdx.y * warpsPerBlock + warpId;  // row‐tile index
    int tileN = blockIdx.x;                           // col‐tile index

    // Guards
    if (tileM >= numTilesM)
                return;
    if (tileN >= numTilesN)
                return;

    int aRow     = tileM * MMA_M;  // row-offset into A/C
    int bCol     = tileN * MMA_N;  // col-offset into B/C

    float accReg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    // reinterpret A and B as uint16_t arrays for bf16 loads
    const uint16_t* A16 = reinterpret_cast<const uint16_t*>(colMajorA_d);
    const uint16_t* B16 = reinterpret_cast<const uint16_t*>(colMajorB_d);

    for (int tileK = 0; tileK < numTilesK; ++tileK) {
        // 1) A‐fragment, reading 4x 32 bit length chunks, each representing 2 bf16 elements from A (column-major):
        int rowBlock  = lane >> 2;          // 0..3
        int colOffset = (lane % 4) * 2;     // {0,2,4,6}
        int i0 = aRow + rowBlock;           // A-row index
        int i1 = aRow + rowBlock + 8;       // A-row+8 index
        int j0 = tileK * MMA_K + colOffset; // A-col index
        int j1 = j0 + 1;                    // next col

        // Column-major: element A[i][j] sits at A16[ j*mDim + i ]
        uint32_t aReg0 = uint32_t(A16[ j0*mDim + i0 ]) | (uint32_t(A16[ j1*mDim + i0 ]) << 16);
        uint32_t aReg1 = uint32_t(A16[ j0*mDim + i1 ]) | (uint32_t(A16[ j1*mDim + i1 ]) << 16);
        uint32_t aReg2 = uint32_t(A16[(j0+8)*mDim + i0]) | (uint32_t(A16[(j1+8)*mDim + i0]) << 16);
        uint32_t aReg3 = uint32_t(A16[(j0+8)*mDim + i1]) | (uint32_t(A16[(j1+8)*mDim + i1]) << 16);


        // 2) B‐fragment, reading  2x 32 bit chunks, each representing 2 bf16 values (column-major):
        int r0 = tileK * MMA_K + (lane % 4) * 2; // B-row
        int r1 = r0 + 1;
        int r2 = r0 + 8;
        int r3 = r2 + 1;
        int c0 = bCol + (lane >> 2);             // B-col

        // Column-major: element B[r][c] sits at B16[ c*kDim + r ]
        uint32_t bReg0 = uint32_t(B16[c0*kDim + r0]) | (uint32_t(B16[c0*kDim + r1]) << 16);
        uint32_t bReg1 = uint32_t(B16[c0*kDim + r2]) | (uint32_t(B16[c0*kDim + r3]) << 16);

        // 3) MMA instruction stays exactly the same (row-col layout):
        // mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        //   {%Rd0, %Rd1, %Rd2, %Rd3},
        //   {%Ra0, %Ra1, %Ra2, %Ra3},
        //   {%Rb0, %Rb1},
        //   {%Rc0, %Rc1, %Rc2, %Rc3};

        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
          : "+f"(accReg[0]), "+f"(accReg[1]),
            "+f"(accReg[2]), "+f"(accReg[3])
          : "r"(aReg0), "r"(aReg1), "r"(aReg2), "r"(aReg3), "r"(bReg0), "r"(bReg1)
        );

    }

    // write back 16×8 float tile into C
    float* cTileDst = C + aRow * nDim + bCol;

    int r = lane / lanesPerSubRow;        // 0..7 for the top half
    int c = (lane % lanesPerSubRow) * 2;    // columns: 0,2,4,6
    cTileDst[r * nDim + c] = accReg[0];
    cTileDst[r * nDim + c + 1] = accReg[1];
    cTileDst[(r + 8) * nDim + c] = accReg[2];
    cTileDst[(r + 8) * nDim + c + 1] = accReg[3];
}