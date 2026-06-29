#include "kernel.h"
#include "helpers.h"

__global__ void k_mmaTensorMatMulM16N8k16Int8(uint8_t *inputMatrixA_d,
                                              uint8_t *inputMatrixB_d,
                                              int *outputMatrix_d,
                                              int mDim,
                                              int nDim,
                                              int kDim) {
    // Tile indices.
    int blockM = blockIdx.y;
    int blockN = blockIdx.x;
    int aRow = blockM * MMA_M; // starting row for this tile in A
    int bCol = blockN * MMA_N; // starting column for this tile in B

    int lane = threadIdx.x % 32;

    // Shared memory:
    // aTile holds the A tile (16x16)
    // bTile holds the B tile (16x8)
    extern __shared__ __align__(16) uint8_t shmem[];
    uint8_t* aTile = shmem;
    uint8_t* bTile = shmem + (MMA_M * MMA_K);

    // Accumulator for MMA (s32) initialized to zero.
    int accReg[4] = {0, 0, 0, 0};

    int numTilesK = kDim / MMA_K;
    for (int tileK = 0; tileK < numTilesK; tileK++) {
        uint8_t* aG = inputMatrixA_d + aRow * kDim + tileK * MMA_K;
        uint8_t* bG = inputMatrixB_d + (tileK * MMA_K) * nDim + bCol;

        // Load A tile into shared memory.
        for (int i = threadIdx.x; i < MMA_M * MMA_K; i += blockDim.x) {
            int r = i / MMA_K;
            int c = i % MMA_K;
            int global_r = aRow + r;
            int global_c = tileK * MMA_K + c;
            aTile[i] = (global_r < mDim && global_c < kDim) ? aG[r * kDim + c] : 0;
        }
        // Load B tile into shared memory
        for (int i = threadIdx.x; i < MMA_K * MMA_N; i += blockDim.x) {
            int r = i / MMA_N;
            int c = i % MMA_N;
            int global_r = tileK * MMA_K + r;
            int global_c = bCol + c;
            bTile[i] = (global_r < kDim && global_c < nDim) ? bG[r * nDim + c] : 0;
        }
        __syncthreads();

        // Load Fragments into registers
        uint32_t fragA[2];
        int offsetA = (lane / 4) * 16 + (lane % 4) * 4;
        manual_load_fragment_A(&aTile[offsetA], fragA, MMA_K);

        int offsetB = 32 * (lane % 4);
        uint32_t fragB = manual_load_fragment_B(&bTile[offsetB]);

        // Perform MMA accumulate.
       asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.u8.u8.s32.satfinite "
                     "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                     : "+r"(accReg[0]), "+r"(accReg[1]), "+r"(accReg[2]), "+r"(accReg[3])
                     : "r"(fragA[0]), "r"(fragA[1]), "r"(fragB));

        __syncthreads();
    }

    // Write the computed 16x8 output tile (from the accumulator) to global memory.
    int* cTile = outputMatrix_d + aRow * nDim + bCol;

    int block_row = lane / 4;  // 0..7
    int block_col = lane % 4;  // 0..3

    int row_out = block_row * 1;
    int col_out = block_col * 2;


    cTile[row_out * nDim + col_out] = accReg[0];
    cTile[row_out * nDim + col_out + 1] = accReg[1];
    cTile[(row_out + 8) * nDim + col_out] = accReg[2];
    cTile[(row_out + 8) * nDim + col_out + 1] = accReg[3];
    __syncthreads();
}