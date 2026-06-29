#include "matrix_mul.h"
#include <cuda_bf16.h>

__global__ void k_mmaM16N8K16AcolBcol(__nv_bfloat16 *colMajorA_d,
                                      __nv_bfloat16 *colMajorB_d,
                                      float *rowMajorC_d,
                                      int mDim, int nDim, int kDim) {
    // Each block handles a 16x8 tile of output C
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int warpId = threadId / 32;
    int laneId = threadId % 32;
    
    // Each warp processes one 16x8 tile of C
    // With 32 threads per warp and 2 warps per block (64 threads), we can handle 2 tiles
    // warpId=0 handles top 16 rows, warpId=1 handles bottom 16 rows of the block's 16 rows
    // Actually for m16n8k16, one warp handles one 16x8 tile
    
    // Thread mapping within warp for m16n8k16:
    // Threads 0-15: handle rows 0-7 of the 16 rows in the tile
    // Threads 16-31: handle rows 8-15 of the 16 rows in the tile
    // Each thread holds 2 elements of A fragment (16x16 = 8 elements per warp, 8 threads hold them)
    // Actually for row.col layout: A fragment is 2x4 = 8 elements, B fragment is 1x8 = 8 elements, C is 2x8 = 16 elements
    
    // Starting position for this warp's tile in C
    int cRow = blockRow * 16 + (warpId / 2) * 16;  // Each warp handles 16 rows
    int cCol = blockCol * 8 + (warpId % 2) * 8;    // Each warp handles 8 cols
    
    // Initialize C accumulator (8 floats for the 2x8 tile, distributed across 4 threads)
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float c4 = 0.0f, c5 = 0.0f, c6 = 0.0f, c7 = 0.0f;
    
    // Loop over k dimension in chunks of 16
    for (int k = 0; k < kDim; k += 16) {
        // Load A fragment (16x16 tile, but we only need 16x8 for this mma)
        // A is column-major: element (i, j) is at index j*mDim + i
        // For mma with row.col, A fragment needs row-major layout in registers
        
        __nv_bfloat16 a0, a1, a2, a3, a4, a5, a6, a7;
        __nv_bfloat16 b0, b1, b2, b3, b4, b5, b6, b7;
        
        // Each thread in the warp loads specific elements
        // For m16n8k16 row.col:
        // A fragment (8 elements): 2 rows x 4 cols
        // B fragment (8 elements): 1 row x 8 cols (but actually it's 8 elements for the k=16 dimension)
        
        // Thread 0-7 in warp load A elements for rows 0-7
        // Thread 8-15 in warp load A elements for rows 8-15
        // Thread 0-7 in warp load B elements
        
        // Calculate row and column indices for this thread's A elements
        int aRow = (warpId % 2) * 8 + (threadId / 4) % 8;  // 0-15 within warp
        int aCol = (k + (threadId % 4) * 4) % 16;          // k offset within 16
        
        // For column-major A: element at (aRow, k + aCol) is at index (k + aCol)*mDim + aRow
        if (aRow < 16 && (k + aCol) < kDim) {
            int aIdx = (k + aCol) * mDim + aRow;
            if (laneId < 16) {
                // Thread loads A element
                __nv_bfloat16 val = colMajorA_d[aIdx];
                // Distribute to appropriate register based on lane
                if (laneId == 0) a0 = val;
                else if (laneId == 1) a1 = val;
                else if (laneId == 2) a2 = val;
                else if (laneId == 3) a3 = val;
                else if (laneId == 4) a4 = val;
                else if (laneId == 5) a5 = val;
                else if (laneId == 6) a6 = val;
                else if (laneId == 7) a7 = val;
            }
        }
        
        // For column-major B: element at (k + bRow, cCol + bCol) is at index (cCol + bCol)*kDim + (k + bRow)
        // B fragment needs 8 elements for the 16 k values
        int bRow = (k + laneId) % 16;
        int bCol = laneId / 16;  // 0 for all 32 threads in warp for this k chunk
        
        if ((k + bRow) < kDim && (cCol + bCol) < nDim) {
            int bIdx = (cCol + bCol) * kDim + (k + bRow);
            __nv_bfloat16 val = colMajorB_d[bIdx];
            // Distribute to appropriate register
            if (laneId < 8) {
                if (laneId == 0) b0 = val;
                else if (laneId == 1) b1 = val;
                else if (laneId == 2) b2 = val;
                else if (laneId == 3) b3 = val;
                else if (laneId == 4) b4 = val;
                else if (laneId == 5) b5 = val;
                else if (laneId == 6) b6 = val;
                else if (laneId == 7) b7 = val;
            }
        }
        
        // Perform mma instruction
        // mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        // C = A * B + C
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3, %4, %5, %6, %7}, "
            "{%8, %9, %10, %11, %12, %13, %14, %15}, "
            "{%16, %17, %18, %19, %20, %21, %22, %23}, "
            "{%24, %25, %26, %27, %28, %29, %30, %31};"
            : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3),
              "=f"(c4), "=f"(c5), "=f"(c6), "=f"(c7)
            : "r"(__cvta_generic_to_int(&a0)), "r"(__cvta_generic_to_int(&a1)),
              "r"(__cvta_generic_to_int(&a2)), "r"(__cvta_generic_to_int(&a3)),
              "r"(__cvta_generic_to_int(&a4)), "r"(__cvta_generic_to_int(&a5)),
              "r"(__cvta_generic_to_int(&a6)), "r"(__cvta_generic_to_int(&a7)),
              "r"(__cvta_generic_to_int(&b0)), "r"(__cvta_generic_to_int(&b1)),
              "r"(__cvta_generic_to_int(&b2)), "r"(__cvta_generic_to_int(&b3)),
              "r"(__cvta_generic_to_int(&b4)), "r"(__cvta_generic_to_int(&b5)),
              "r"(__cvta_generic_to_int(&b6)), "r"(__cvta_generic_to_int(&b7)),
              "f"(c0), "f"(c1), "f"(c2), "f"(c3),
              "f"(c4), "f"(c5), "f"(c6), "f"(c7)
        );
    }
    
    // Store C fragment to row-major output
    // C fragment layout: 2 rows x 8 cols, stored in c0-c7
    // c0-c3: row 0, cols 0-3
    // c4-c7: row 1, cols 0-3
    // But we need to handle the full 16 rows from warp perspective
    
    int localRow = (warpId % 2) * 8 + (threadId / 4) % 8;
    int localCol = (threadId % 4);
    
    if (localRow < 16 && localCol < 8) {
        int outRow = cRow + localRow;
        int outCol = cCol + localCol;
        if (outRow < mDim && outCol < nDim) {
            int fragIdx = (localRow % 8) * 4 + localCol;
            float val = (fragIdx < 4) ? c_frag[fragIdx] : c_frag[fragIdx];
            rowMajorC_d[outRow * nDim + outCol] = val;
        }
    }
}