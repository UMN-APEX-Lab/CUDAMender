#include <cstdio>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <random>
#include <cuda_runtime.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>
#include "mma_kernel.h"

#define CUDA_CHECK(call)                                                           \
do {                                                                               \
        cudaError_t error = call;                                                  \
        if (error != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),\
                    __FILE__, __LINE__);                                           \
            exit(error);                                                           \
        }                                                                          \
} while (0)
#undef NDEBUG

// Function to compute valid reference result
// Inputs => ( Row Major -> A, Column Major -> B)
void cpuMatMulReference(const __nv_bfloat16* A,
                        const __nv_bfloat16* B,
                        float* cpuRefC,
                        int M,
                        int N,
                        int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = static_cast<float>(A[i*K + k]);
                float b_val = static_cast<float>(B[j*K + k]);
                sum += a_val * b_val;
            }
            cpuRefC[i*N + j] = sum;
        }
    }
}

void launch() {

    const int TEST_CASE_COUNT = 7;
    const int PROBLEM_DIMS = 3;

    //Test case dimensions {M, N, K}
    const int MAX_M = 512;
    const int MAX_N = 512;
    const int MAX_K = 512;

    const int TEST_CASES_DIMS[TEST_CASE_COUNT][PROBLEM_DIMS] = {{16,8,16}, {512,512,512}, {32,16,32}, {256, 256, 256}, {64, 64, 64} , {64, 32, 32}, {128, 128, 128}};

    // Tolerance for validation, set to 1% due to nature of half precision operations
    const float TOLERANCE  = 0.01;

    int deviceId = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    const int warpSize = prop.warpSize;    // 32

    // Choose 4 warps per block (arbitrary tuning; can be tuned further)
    const int WARPS_PER_BLOCK = 4;
    const int BLOCK_SIZE      = warpSize * WARPS_PER_BLOCK;  // 128 threads


    //Set up random number generation with fixed seed for reproducibility
    std::mt19937 randEngine(42); // Fixed seed
    // Bounded random distribution for test case initialization
    std::uniform_real_distribution<float> randDist(1.0f, 100.0f);

    // Use a CUDA stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    //Pointers for Host Memory
    __nv_bfloat16* A_h =(__nv_bfloat16*)malloc(MAX_M * MAX_K * sizeof(__nv_bfloat16));
    __nv_bfloat16* B_h =(__nv_bfloat16*)malloc(MAX_K * MAX_N * sizeof(__nv_bfloat16));

    float* cpuC_h =(float*)malloc(MAX_M * MAX_N * sizeof(float)); // Reference Matrix space allocation on host
    float* gpuC_h = (float*)malloc(MAX_M * MAX_N * sizeof(float));// GPU result Matrix space allocation on host

    //Pointers for device memory (GPU)
    __nv_bfloat16* A_d;
    __nv_bfloat16* B_d;
    float* C_d;

    // Allocate the memory on the device
    CUDA_CHECK(cudaMallocAsync(&A_d, MAX_M * MAX_K * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&B_d, MAX_K * MAX_N * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&C_d, MAX_M * MAX_N * sizeof(float), stream));

    for (int i = 0; i < TEST_CASE_COUNT; i++) {
        // Dimensions of the input and output layers
        int M = TEST_CASES_DIMS[i][0]; //Number of Rows in Matrix A
        int N = TEST_CASES_DIMS[i][1]; //Number of Columns in Matrix B
        int K = TEST_CASES_DIMS[i][2]; //Number of Columns in Matrix A and Rows in Matrix B

        //Populating input matrices with random values

        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < K ; ++c) {
                float val = randDist(randEngine);
                A_h[r * K + c] = __nv_bfloat16(val); // Filling A Matrix in Row Major Way
            }
       }

       for (int c = 0; c < N; ++c) {
            for (int r = 0; r < K; ++r) {
                float  val = randDist(randEngine);   // Filling B Matrix in Row Major Way
                B_h[c * K + r] = __nv_bfloat16(val);
            }
        }

        // Initialize the result on the device
        CUDA_CHECK(cudaMemsetAsync(C_d, 0, M * N * sizeof(float), stream));

        // Load Test Cases
        CUDA_CHECK(cudaMemcpyAsync(A_d, A_h, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(B_d, B_h, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));

        // Check if the dimensions are divisible by the block tile dimensions
        assert(M % MMA_M == 0);
        assert(N % MMA_N == 0);
        assert(K % MMA_K == 0);

        // Compute tile counts and grid dimensions
        int numTilesM = (M + MMA_M - 1)/ MMA_M;  // how many 16‐row blocks
        int numTilesN = (N + MMA_N - 1)/ MMA_N;  // how many 8‐col blocks

        int gridY = (numTilesM + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;  // ceil(numTilesM / WARPS_PER_BLOCK)
        int gridX = numTilesN;  // one block per N‐tile, warps carve M‐tiles

        dim3 blockDim(BLOCK_SIZE, 1, 1);
        dim3 gridDim(gridX, gridY);

        // Shared‐memory per block
        size_t shmemBytes = WARPS_PER_BLOCK * (MMA_M*MMA_K + MMA_K*MMA_N)
                           * sizeof(__nv_bfloat16);

        // Launch kernel
        void *args[] = {&A_d,
                        &B_d,
                        &C_d,
                        (void*)&M,
                        (void*)&N,
                        (void*)&K};

        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaM16N8K16ArowBcol,
                                    gridDim,
                                    blockDim,
                                    args,
                                    shmemBytes,
                                    stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        cpuMatMulReference(A_h, B_h, cpuC_h, M, N, K);

        // Copying the result back to the host
        CUDA_CHECK(cudaMemcpyAsync(gpuC_h, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream));

        // Validate the result, with in 1% tolerance
        for(int t = 0; t < M*N; ++t) {
            assert(std::fabs((gpuC_h[t] - cpuC_h[t]) / std::fabs(cpuC_h[t])) <= TOLERANCE);
        }

   }
    // Free up resources
    CUDA_CHECK(cudaFreeAsync(A_d, stream));
    CUDA_CHECK(cudaFreeAsync(B_d, stream));
    CUDA_CHECK(cudaFreeAsync(C_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(A_h);
    free(B_h);
    free(cpuC_h);
    free(gpuC_h);
}

void benchmark() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const int WARMUP_ITERS = 3;
    const int TIMED_ITERS = 100;

    int deviceId = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
    const int warpSize = prop.warpSize;

    const int WARPS_PER_BLOCK = 4;
    const int BLOCK_SIZE = warpSize * WARPS_PER_BLOCK;

    std::mt19937 randEngine(123);
    std::uniform_real_distribution<float> randDist(1.0f, 100.0f);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    __nv_bfloat16* A_h = (__nv_bfloat16*)malloc(M * K * sizeof(__nv_bfloat16));
    __nv_bfloat16* B_h = (__nv_bfloat16*)malloc(K * N * sizeof(__nv_bfloat16));

    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c)
            A_h[r * K + c] = __nv_bfloat16(randDist(randEngine));

    for (int c = 0; c < N; ++c)
        for (int r = 0; r < K; ++r)
            B_h[c * K + r] = __nv_bfloat16(randDist(randEngine));

    __nv_bfloat16* A_d;
    __nv_bfloat16* B_d;
    float* C_d;

    CUDA_CHECK(cudaMallocAsync(&A_d, M * K * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&B_d, K * N * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&C_d, M * N * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(A_d, A_h, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(B_d, B_h, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));

    int numTilesM = (M + MMA_M - 1) / MMA_M;
    int numTilesN = (N + MMA_N - 1) / MMA_N;

    int gridY = (numTilesM + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int gridX = numTilesN;

    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(gridX, gridY);

    size_t shmemBytes = WARPS_PER_BLOCK * (MMA_M * MMA_K + MMA_K * MMA_N)
                        * sizeof(__nv_bfloat16);

    void* args[] = {&A_d, &B_d, &C_d,
                    (void*)&M, (void*)&N, (void*)&K};

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaMemsetAsync(C_d, 0, M * N * sizeof(float), stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaM16N8K16ArowBcol,
                                    gridDim, blockDim, args, shmemBytes, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; ++i) {
        CUDA_CHECK(cudaMemsetAsync(C_d, 0, M * N * sizeof(float), stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaM16N8K16ArowBcol,
                                    gridDim, blockDim, args, shmemBytes, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(A_d, stream));
    CUDA_CHECK(cudaFreeAsync(B_d, stream));
    CUDA_CHECK(cudaFreeAsync(C_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(A_h);
    free(B_h);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}