#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

// Tile dimensions for GEMM
#define GEMM_BLOCK_M 64
#define GEMM_BLOCK_N 64
#define GEMM_BLOCK_K 32

// Softmax block size
#define SOFTMAX_BLOCK 256

// GEMM kernel: D = alpha * A * B + beta * C
// A: row-major [M, K], B: column-major [K, N], C/D: row-major [M, N]
__global__ void gemm_kernel(
    int M, int N, int K,
    float alpha, float beta,
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    const __half* __restrict__ C, int ldc,
    __half* __restrict__ D, int ldd,
    int64_t batch_stride_A,
    int64_t batch_stride_B,
    int64_t batch_stride_C,
    int64_t batch_stride_D) {
    
    // Block indices
    int bx = blockIdx.x; // N dimension
    int by = blockIdx.y; // M dimension
    int batch = blockIdx.z;
    
    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Warp size is 32, so we have (GEMM_BLOCK_M/8) * (GEMM_BLOCK_N/8) warps
    // Each thread computes an 8x8 tile
    
    const int THREAD_M = 8;
    const int THREAD_N = 8;
    
    int warpId = (ty * (GEMM_BLOCK_N / THREAD_N) + tx) / 32;
    int laneId = (ty * (GEMM_BLOCK_N / THREAD_N) + tx) % 32;
    
    // Each warp computes a sub-tile
    int warpM = warpId / (GEMM_BLOCK_N / 32);
    int warpN = warpId % (GEMM_BLOCK_N / 32);
    
    // Thread within warp computes 8x8
    int threadM = (laneId / 4) * THREAD_M;
    int threadN = (laneId % 4) * THREAD_N;
    
    int mBase = by * GEMM_BLOCK_M + warpM * 16 + threadM;
    int nBase = bx * GEMM_BLOCK_N + warpN * 32 + threadN;
    
    // Batch offsets
    const __half* A_batch = A + batch * batch_stride_A;
    const __half* B_batch = B + batch * batch_stride_B;
    const __half* C_batch = C + batch * batch_stride_C;
    __half* D_batch = D + batch * batch_stride_D;
    
    // Accumulators in FP32
    float acc[THREAD_M][THREAD_N];
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Shared memory for A and B tiles
    __shared__ __half sA[GEMM_BLOCK_M][GEMM_BLOCK_K + 2];  // +2 for bank conflict avoidance
    __shared__ __half sB[GEMM_BLOCK_K][GEMM_BLOCK_N + 2];
    
    // Loop over K tiles
    for (int kTile = 0; kTile < K; kTile += GEMM_BLOCK_K) {
        // Load A tile (row-major) into shared memory
        // Each thread loads multiple elements
        #pragma unroll
        for (int i = 0; i < GEMM_BLOCK_M; i += (GEMM_BLOCK_M / (GEMM_BLOCK_K / 4))) {
            int loadM = ty * (GEMM_BLOCK_M / (GEMM_BLOCK_K / 4)) + i;
            int loadK = tx * 4;
            if (loadM < GEMM_BLOCK_M && kTile + loadK < K) {
                int globalM = by * GEMM_BLOCK_M + loadM;
                if (globalM < M) {
                    // Load 4 halfs
                    #pragma unroll
                    for (int e = 0; e < 4 && kTile + loadK + e < K; e++) {
                        sA[loadM][loadK + e] = A_batch[globalM * lda + kTile + loadK + e];
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        sA[loadM][loadK + e] = __float2half(0.0f);
                    }
                }
            }
        }
        
        // Load B tile (column-major) into shared memory
        #pragma unroll
        for (int i = 0; i < GEMM_BLOCK_K; i += (GEMM_BLOCK_K / (GEMM_BLOCK_N / 4))) {
            int loadK = ty * (GEMM_BLOCK_K / (GEMM_BLOCK_N / 4)) + i;
            int loadN = tx * 4;
            if (loadK < GEMM_BLOCK_K && kTile + loadK < K) {
                int globalN = bx * GEMM_BLOCK_N + loadN;
                if (globalN < N) {
                    #pragma unroll
                    for (int e = 0; e < 4 && globalN + e < N; e++) {
                        // B is column-major: B[k, n] = B[k + n * ldb]
                        sB[loadK][loadN + e] = B_batch[(kTile + loadK) + (globalN + e) * ldb];
                    }
                } else {
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        sB[loadK][loadN + e] = __float2half(0.0f);
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Compute on the tile
        #pragma unroll
        for (int k = 0; k < GEMM_BLOCK_K && kTile + k < K; k++) {
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                int mIdx = warpM * 16 + threadM + i;
                float aVal = __half2float(sA[mIdx][k]);
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    int nIdx = warpN * 32 + threadN + j;
                    float bVal = __half2float(sB[k][nIdx]);
                    acc[i][j] += aVal * bVal;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output with alpha and beta
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        int globalM = mBase + i;
        if (globalM >= M) break;
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int globalN = nBase + j;
            if (globalN >= N) break;
            
            float result = alpha * acc[i][j];
            if (beta != 0.0f) {
                result += beta * __half2float(C_batch[globalM * ldc + globalN]);
            }
            D_batch[globalM * ldd + globalN] = __float2half(result);
        }
    }
}

// Simpler GEMM kernel using 1D thread blocks for better occupancy
__global__ void gemm_kernel_simple(
    int M, int N, int K,
    float alpha, float beta,
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    const __half* __restrict__ C, int ldc,
    __half* __restrict__ D, int ldd,
    int64_t batch_stride_A,
    int64_t batch_stride_B,
    int64_t batch_stride_C,
    int64_t batch_stride_D) {
    
    int batch = blockIdx.z;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (m >= M || n >= N) return;
    
    const __half* A_batch = A + batch * batch_stride_A;
    const __half* B_batch = B + batch * batch_stride_B;
    const __half* C_batch = C + batch * batch_stride_C;
    __half* D_batch = D + batch * batch_stride_D;
    
    // Compute dot product in FP32
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = __half2float(A_batch[m * lda + k]);
        float b = __half2float(B_batch[k + n * ldb]);
        sum += a * b;
    }
    
    float result = alpha * sum;
    if (beta != 0.0f) {
        result += beta * __half2float(C_batch[m * ldc + n]);
    }
    
    D_batch[m * ldd + n] = __float2half(result);
}

// Optimized GEMM kernel with better shared memory usage
template<int BM, int BN, int BK>
__global__ void gemm_kernel_opt(
    int M, int N, int K,
    float alpha, float beta,
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    const __half* __restrict__ C, int ldc,
    __half* __restrict__ D, int ldd,
    int64_t batch_stride_A,
    int64_t batch_stride_B,
    int64_t batch_stride_C,
    int64_t batch_stride_D) {
    
    const int TM = 8;  // Threads per M dimension in block
    const int TN = 8;  // Threads per N dimension in block
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int batch = blockIdx.z;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Tile in global memory
    int mStart = by * BM;
    int nStart = bx * BN;
    
    // Thread's position within tile
    int mIdx = ty * TM;
    int nIdx = tx * TN;
    
    // Batch pointers
    const __half* A_batch = A + batch * batch_stride_A;
    const __half* B_batch = B + batch * batch_stride_B;
    const __half* C_batch = C + batch * batch_stride_C;
    __half* D_batch = D + batch * batch_stride_D;
    
    // Shared memory
    __shared__ __half sA[BM][BK + 2];
    __shared__ __half sB[BK][BN + 2];
    
    // Accumulators
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Loop over K
    for (int kTile = 0; kTile < K; kTile += BK) {
        // Load A: row-major [M, K]
        // Each thread loads TM elements from A
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            int globalM = mStart + mIdx + i;
            int globalK = kTile + tx;
            if (globalM < M && globalK < K) {
                sA[mIdx + i][tx] = A_batch[globalM * lda + globalK];
            } else {
                sA[mIdx + i][tx] = __float2half(0.0f);
            }
        }
        
        // Load B: column-major [K, N]
        // Each thread loads TN elements from B
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int globalK = kTile + ty;
            int globalN = nStart + nIdx + j;
            if (globalK < K && globalN < N) {
                sB[ty][nIdx + j] = B_batch[globalK + globalN * ldb];
            } else {
                sB[ty][nIdx + j] = __float2half(0.0f);
            }
        }
        
        __syncthreads();
        
        // Compute
        #pragma unroll
        for (int k = 0; k < BK && kTile + k < K; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                float aVal = __half2float(sA[mIdx + i][k]);
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    float bVal = __half2float(sB[k][nIdx + j]);
                    acc[i][j] += aVal * bVal;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store with alpha/beta
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int globalM = mStart + mIdx + i;
        if (globalM >= M) continue;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int globalN = nStart + nIdx + j;
            if (globalN >= N) continue;
            
            float result = alpha * acc[i][j];
            if (beta != 0.0f) {
                result += beta * __half2float(C_batch[globalM * ldc + globalN]);
            }
            D_batch[globalM * ldd + globalN] = __float2half(result);
        }
    }
}

// Softmax kernel: row-wise softmax on D
// Each row is processed by a block of threads
__global__ void softmax_kernel(
    int M, int N,
    const __half* __restrict__ D, int ldd,
    __half* __restrict__ Softmax, int lds,
    int64_t batch_stride_D,
    int64_t batch_stride_Softmax) {
    
    int batch = blockIdx.y;
    int m = blockIdx.x;
    
    const __half* D_batch = D + batch * batch_stride_D;
    __half* Softmax_batch = Softmax + batch * batch_stride_Softmax;
    
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    
    // Shared memory for reduction
    __shared__ float sMax[32];
    __shared__ float sSum[32];
    __shared__ float rowMax;
    __shared__ float rowSum;
    
    // Step 1: Find max per row
    float localMax = -FLT_MAX;
    
    // Each thread finds local max
    for (int n = tid; n < N; n += blockDim.x) {
        float val = __half2float(D_batch[m * ldd + n]);
        localMax = fmaxf(localMax, val);
    }
    
    // Warp reduction for max
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        localMax = fmaxf(localMax, __shfl_down_sync(0xffffffff, localMax, offset));
    }
    
    if (lane == 0) {
        sMax[warp] = localMax;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < 32) {
        float warpMax = (tid < (blockDim.x + 31) / 32) ? sMax[tid] : -FLT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warpMax = fmaxf(warpMax, __shfl_down_sync(0xffffffff, warpMax, offset));
        }
        if (tid == 0) {
            rowMax = warpMax;
        }
    }
    __syncthreads();
    
    float maxVal = rowMax;
    
    // Step 2: Compute exp and sum
    float localSum = 0.0f;
    // Store exp values in registers or recompute? Recompute to save memory
    // Actually, we need to store for second pass. Use shared memory if N is small,
    // otherwise recompute. For typical cases, recompute is fine.
    
    // First pass: compute local sum of exp(x - max)
    for (int n = tid; n < N; n += blockDim.x) {
        float val = __half2float(D_batch[m * ldd + n]);
        localSum += expf(val - maxVal);
    }
    
    // Warp reduction for sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    }
    
    if (lane == 0) {
        sSum[warp] = localSum;
    }
    __syncthreads();
    
    // Final reduction
    if (tid < 32) {
        float warpSum = (tid < (blockDim.x + 31) / 32) ? sSum[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }
        if (tid == 0) {
            rowSum = warpSum;
        }
    }
    __syncthreads();
    
    float sumVal = rowSum;
    
    // Step 3: Compute final softmax
    for (int n = tid; n < N; n += blockDim.x) {
        float val = __half2float(D_batch[m * ldd + n]);
        float softmaxVal = expf(val - maxVal) / sumVal;
        Softmax_batch[m * lds + n] = __float2half(softmaxVal);
    }
}

// Optimized softmax using vectorized loads
__global__ void softmax_kernel_opt(
    int M, int N,
    const __half* __restrict__ D, int ldd,
    __half* __restrict__ Softmax, int lds,
    int64_t batch_stride_D,
    int64_t batch_stride_Softmax) {
    
    int batch = blockIdx.y;
    int m = blockIdx.x;
    
    const __half* D_row = D + batch * batch_stride_D + m * ldd;
    __half* Softmax_row = Softmax + batch * batch_stride_Softmax + m * lds;
    
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int numWarps = (blockDim.x + 31) >> 5;
    
    __shared__ float sMax[32];
    __shared__ float sSum[32];
    __shared__ float rowMax;
    __shared__ float rowSum;
    
    // Find max
    float localMax = -1e30f;
    for (int n = tid; n < N; n += blockDim.x) {
        float val = __half2float(D_row[n]);
        localMax = fmaxf(localMax, val);
    }
    
    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        localMax = fmaxf(localMax, __shfl_down_sync(0xffffffff, localMax, offset));
    }
    if (lane == 0) sMax[warp] = localMax;
    __syncthreads();
    
    if (tid < 32) {
        float warpMax = (tid < numWarps) ? sMax[tid] : -1e30f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warpMax = fmaxf(warpMax, __shfl_down_sync(0xffffffff, warpMax, offset));
        }
        if (tid == 0) rowMax = warpMax;
    }
    __syncthreads();
    
    float maxVal = rowMax;
    
    // Compute sum of exp
    float localSum = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        localSum += expf(__half2float(D_row[n]) - maxVal);
    }
    
    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    }
    if (lane == 0) sSum[warp] = localSum;
    __syncthreads();
    
    if (tid < 32) {
        float warpSum = (tid < numWarps) ? sSum[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warpSum += __shfl_down_sync(0xffffffff, warpSum, offset);
        }
        if (tid == 0) rowSum = warpSum;
    }
    __syncthreads();
    
    float sumVal = rowSum;
    
    // Write output
    for (int n = tid; n < N; n += blockDim.x) {
        float val = __half2float(D_row[n]);
        float result = expf(val - maxVal) / sumVal;
        Softmax_row[n] = __float2half(result);
    }
}

// Fused GEMM + Softmax kernel for small sizes
// This computes GEMM and softmax in one kernel to reduce memory traffic
__global__ void fused_gemm_softmax_kernel(
    int M, int N, int K,
    float alpha, float beta,
    const __half* __restrict__ A, int lda,
    const __half* __restrict__ B, int ldb,
    const __half* __restrict__ C, int ldc,
    __half* __restrict__ Softmax, int lds,
    int64_t batch_stride_A,
    int64_t batch_stride_B,
    int64_t batch_stride_C,
    int64_t batch_stride_Softmax) {
    
    int batch = blockIdx.z;
    int m = blockIdx.y;
    
    const __half* A_batch = A + batch * batch_stride_A;
    const __half* B_batch = B + batch * batch_stride_B;
    const __half* C_batch = C + batch * batch_stride_C;
    __half* Softmax_batch = Softmax + batch * batch_stride_Softmax;
    
    int tid = threadIdx.x;
    
    // Compute GEMM for this row
    // Each thread computes some columns
    // Store in registers first, then do softmax
    
    // For small N, we can store in registers
    // For larger N, we need to tile
    
    // Simple approach: compute one element per thread, multiple passes for softmax
    // Actually, let's compute all elements for this row in shared memory
    
    extern __shared__ float sRow[];
    float* rowValues = sRow; // Size: N elements
    
    // Initialize
    for (int n = tid; n < N; n += blockDim.x) {
        rowValues[n] = 0.0f;
    }
    __syncthreads();
    
    // Compute GEMM: each thread computes partial sum for its assigned columns
    // But we need all of A[m,:] dotted with B[:,n] for each n
    
    // Better: each thread handles a subset of n values
    for (int n = tid; n < N; n += blockDim.x) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = __half2float(A_batch[m * lda + k]);
            float b = __half2float(B_batch[k + n * ldb]);
            sum += a * b;
        }
        float result = alpha * sum;
        if (beta != 0.0f) {
            result += beta * __half2float(C_batch[m * ldc + n]);
        }
        rowValues[n] = result;
    }
    __syncthreads();
    
    // Now do softmax on rowValues
    // Step 1: find max
    float localMax = -FLT_MAX;
    for (int n = tid; n < N; n += blockDim.x) {
        localMax = fmaxf(localMax, rowValues[n]);
    }
    
    // Reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        localMax = fmaxf(localMax, __shfl_down_sync(0xffffffff, localMax, offset));
    }
    
    // Broadcast to all threads in warp
    localMax = __shfl_sync(0xffffffff, localMax, 0);
    
    // Step 2: compute exp and sum
    float localSum = 0.0f;
    for (int n = tid; n < N; n += blockDim.x) {
        float expVal = expf(rowValues[n] - localMax);
        rowValues[n] = expVal;  // Store exp value
        localSum += expVal;
    }
    
    // Reduce sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        localSum += __shfl_down_sync(0xffffffff, localSum, offset);
    }
    localSum = __shfl_sync(0xffffffff, localSum, 0);
    
    // Step 3: normalize and write
    for (int n = tid; n < N; n += blockDim.x) {
        float softmaxVal = rowValues[n] / localSum;
        Softmax_batch[m * lds + n] = __float2half(softmaxVal);
    }
}

cudaError_t gemm_softmax_solution(
    int M, int N, int K,
    float alpha, float beta,
    __half const *A, int lda,
    __half const *B, int ldb,
    __half const *C, int ldc,
    __half *D, int ldd,
    __half *Softmax, int lds,
    int batch_count,
    int64_t batch_stride_A,
    int64_t batch_stride_B,
    int64_t batch_stride_C,
    int64_t batch_stride_D,
    int64_t batch_stride_Softmax) {
    
    cudaError_t err = cudaSuccess;
    
    // Choose strategy based on problem size
    
    // For small problems, use fused kernel
    // For larger problems, use separate kernels with optimized GEMM
    
    // Threshold for fused kernel
    const int FUSED_MAX_N = 2048;
    const int FUSED_MAX_K = 512;
    
    bool useFused = (N <= FUSED_MAX_N && K <= FUSED_MAX_K && M <= 256);
    
    if (useFused) {
        // Fused kernel: one block per row
        int threads = 256;
        if (N < 128) threads = 128;
        if (N < 64) threads = 64;
        
        dim3 grid(M, batch_count);
        size_t smemSize = N * sizeof(float);
        
        fused_gemm_softmax_kernel<<<grid, threads, smemSize>>>(
            M, N, K, alpha, beta,
            A, lda, B, ldb, C, ldc, Softmax, lds,
            batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_Softmax);
        
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        
        return cudaSuccess;
    }
    
    // Separate GEMM and Softmax kernels
    
    // GEMM configuration
    // Use tile-based kernel for better performance
    const int BM = 64;
    const int BN = 64;
    const int BK = 32;
    
    dim3 gemmGrid((N + BN - 1) / BN, (M + BM - 1) / BM, batch_count);
    dim3 gemmBlock(BN / 8, BM / 8);  // Each thread computes 8x8
    
    // For very small sizes, use simple kernel
    if (M <= 64 && N <= 64) {
        dim3 simpleBlock(16, 16);
        dim3 simpleGrid((N + 15) / 16, (M + 15) / 16, batch_count);
        
        gemm_kernel_simple<<<simpleGrid, simpleBlock>>>(
            M, N, K, alpha, beta,
            A, lda, B, ldb, C, ldc, D, ldd,
            batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D);
    } else {
        // Use optimized kernel
        // Adjust tile size based on problem
        if (M >= 256 && N >= 256) {
            // Larger tiles for bigger matrices
            dim3 block(8, 8);  // 64 threads, each handles 8x8
            dim3 grid((N + 63) / 64, (M + 63) / 64, batch_count);
            
            gemm_kernel_opt<64, 64, 32><<<grid, block>>>(
                M, N, K, alpha, beta,
                A, lda, B, ldb, C, ldc, D, ldd,
                batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D);
        } else {
            // Smaller tiles
            dim3 block(8, 8);
            dim3 grid((N + 63) / 64, (M + 63) / 64, batch_count);
            
            gemm_kernel_opt<64, 64, 32><<<grid, block>>>(
                M, N, K, alpha, beta,
                A, lda, B, ldb, C, ldc, D, ldd,
                batch_stride_A, batch_stride_B, batch_stride_C, batch_stride_D);
        }
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    
    // Softmax kernel
    // One block per row, multiple batches
    int softmaxThreads = 256;
    if (N <= 128) softmaxThreads = 128;
    if (N <= 64) softmaxThreads = 64;
    
    dim3 softmaxGrid(M, batch_count);
    
    softmax_kernel_opt<<<softmaxGrid, softmaxThreads>>>(
        M, N, D, ldd, Softmax, lds,
        batch_stride_D, batch_stride_Softmax);
    
    err = cudaGetLastError();
    return err;
}
