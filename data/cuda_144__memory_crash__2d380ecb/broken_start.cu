#include "kernel.h"
#include "helpers.h"

__global__ void k_mmaTensorMatMulM16N8k16Int8(uint8_t *inputMatrixA_d, 
                                              uint8_t *inputMatrixB_d, 
                                              int *outputMatrix_d, 
                                              int mDim, 
                                              int nDim, 
                                              int kDim) {
    // The mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 instruction
    // consumes fragments distributed across a warp of 32 threads.
    
    // Matrix A: m16 x k16, Row Major. 
    // Each thread needs 2 registers of 32-bit (containing 4 int8 each).
    uint32_t fragA[2];
    
    // Matrix B: k16 x n8, Col Major.
    // Each thread needs 1 register of 32-bit (containing 4 int8).
    uint32_t fragB;
    
    // Matrix C: m16 x n8, Row Major.
    // Each thread needs 4 registers of 32-bit (int32).
    int fragC[4] = {0, 0, 0, 0};

    // Calculate block indices
    // Since the MMA instruction handles a fixed 16x8x16 tile, 
    // and the problem states compatible dimensions, we calculate global offsets.
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x / 32) * MMA_M;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.x % 1) * MMA_N; // Simplified for single warp case or specific grid
    // However, for a general case based on problem constraints:
    int m_offset = blockIdx.x * MMA_M;
    int n_offset = blockIdx.y * MMA_N;

    for (int k = 0; k < kDim; k += MMA_K) {
        // Load fragment A
        // Thread i in warp handles specific rows of the 16x16 tile.
        // Based on mma.sync.aligned.m16n8k16 layout:
        // Thread 0-15: fragA[0] = row i, col 0-3; fragA[1] = row i, col 8-11
        // Thread 16-31: fragA[0] = row i-16, col 0-3; fragA[1] = row i-16, col 8-11
        // Actually, for m16n8k16, threads 0-15 and 16-31 both contribute.
        int lane = threadIdx.x % 32;
        int row = lane % 16;
        
        const uint8_t* a_ptr = inputMatrixA_d + (m_offset + row) * kDim + k;
        // manual_load_fragment_A uses src[0-3] and src[8*stride...]. 
        // stride is the row width (kDim).
        manual_load_fragment_A(a_ptr, fragA, kDim);

        // Load fragment B
        // Matrix B is column major. k16 x n8.
        const uint8_t* b_ptr = inputMatrixB_d + n_offset * MMA_K + k;
        fragB = manual_load_fragment_B(b_ptr);

        // PTX MMA Instruction
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};"
            : "=&r"(fragC[0]), "=&r"(fragC[1]), "=&r"(fragC[2]), "=&r"(fragC[3])
            : "r"(fragA[0]), "r"(fragA[1]), 
              "r"(fragB), 
              "r"(fragC[0]), "r"(fragC[1]), "r"(fragC[2]), "r"(fragC[3])
        );
    }

    // Store C
    // The distribution of C in registers for m16n8k16:
    // Thread lane handles specific elements of the 16x8 result.
    // Lane i (0-15) handles C[i, 0], C[i, 1]... no, it's more complex.
    // According to NVIDIA docs for m16n8k16:
    // Thread i (0-15) owns: C[i, 0], C[i, 2], C[i, 4], C[i, 6] (if we look at the 4 registers)
    // Actually, for m16n8k16, each thread owns 4 elements.
    // Thread i (0-15) owns: C[i, 0], C[i, 1], C[i, 2], C[i, 3] is not correct.
    // Correct mapping for m16n8k16:
    // Thread i (0-15) owns: C[i, 0], C[i, 2], C[i, 4], C[i, 6] 
    // Thread i (16-31) owns: C[i-16, 1], C[i-16, 3], C[i-16, 5], C[i-16, 7]
    
    int lane = threadIdx.x % 32;
    if (lane < 16) {
        int row = lane;
        outputMatrix_d[(m_offset + row) * nDim + (n_offset + 0)] = fragC[0];
        outputMatrix_d[(m_offset + row) * nDim + (n_offset + 2)] = fragC[1];
        outputMatrix_d[(m_offset + row) * nDim + (n_offset + 4)] = fragC[2];
        outputMatrix_d[(m_offset + row) * nDim + (n_offset + 6)] = fragC[3];
    } else {
        int row = lane - 16;
        outputMatrix_d[(m_offset + row) * nDim + (n_offset + 1)] = fragC[0];
        outputMatrix_d[(m_offset + row) * nDim + (n_offset + 3)] = fragC[1];
        outputMatrix_d[(m_offset + row) * nDim + (n_offset + 5)] = fragC[2];
        outputMatrix_d[(m_offset + row) * nDim + (n_offset + 7)] = fragC[3];
    }
}
