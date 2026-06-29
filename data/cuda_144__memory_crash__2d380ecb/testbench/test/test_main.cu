#include <cassert>
#include <cstring>
#include <random>
#include <cuda_runtime.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>
#include "kernel.h"
#include "helpers.h"

#undef NDEBUG

void cpuMatMulReference(const uint8_t* A,
                        const uint8_t* B,
                        int* cpuRefC,
                        int M,
                        int N,
                        int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                int a_val = static_cast<int>(A[i * K + k]);
                int b_val = static_cast<int>(B[k * N + j]);
                sum += a_val * b_val;
            }
            cpuRefC[i * N + j] = sum;
        }
    }
}

void launch() {
    const int TEST_CASE_COUNT = 7;
    //Test case dimensions {M, N, K}
    const int TEST_CASES_DIMS[TEST_CASE_COUNT][3] = {{16,8,16}, {512,512,512}, {32,16,32}, {256, 256, 256}, {64, 64, 64} , {64, 32, 32}, {128, 128, 128}};

    const int BLOCK_SIZE = 256;

    //Set up random number generation with fixed seed for reproducibility
    std::mt19937 randEngine(42); // Fixed seed

    // Bounded random distribution for test case initialization
    std::uniform_real_distribution<float> randDist(1.0f, 255.0f);

    for (int i = 0; i < TEST_CASE_COUNT; i++) {
        // Dimensions of the input and output layers
        int M = TEST_CASES_DIMS[i][0]; //Number of Rows in Matrix A
        int N = TEST_CASES_DIMS[i][1]; //Number of Columns in Matrix B
        int K = TEST_CASES_DIMS[i][2]; //Number of Columns in Matrix A and Rows in Matrix B

        //Pointers for Host Memory
        uint8_t* A_h =(uint8_t*)malloc(M * K * sizeof(uint8_t));
        uint8_t* B_h =(uint8_t*)malloc(K * N * sizeof(uint8_t));

        int* cpuC_h =(int*)malloc(M * N * sizeof(int)); // Reference Matrix space allocation on host
        int* gpuC_h = (int*)malloc(M * N * sizeof(int));// GPU result Matrix space allocation on host

        //Pointers for device memory (GPU)
        uint8_t* A_d;
        uint8_t* B_d;
        int* C_d;

        //Populating input matrices with random values
        for (int i = 0; i < M * K; i++) {
            uint32_t val = randDist(randEngine);
            A_h[i] = uint8_t(val);
        }

        for (int i = 0; i < K * N; i++) {
            uint32_t val = randDist(randEngine);
            B_h[i] = uint8_t(val);
        }

        // Use a CUDA stream for asynchronous operations
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Allocate the memory on the device
        CUDA_CHECK(cudaMallocAsync(&A_d, M * K * sizeof(uint8_t), stream));
        CUDA_CHECK(cudaMallocAsync(&B_d, K * N * sizeof(uint8_t), stream));
        CUDA_CHECK(cudaMallocAsync(&C_d, M * N * sizeof(int), stream));

        //Load Test Cases
        CUDA_CHECK(cudaMemcpyAsync(A_d, A_h, M * K * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(B_d, B_h, K * N * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

        // Initialize the result on the device
        CUDA_CHECK(cudaMemsetAsync(C_d, 0, M * N * sizeof(int), stream));

        //Check if the dimensions are divisible by the block tile dimensions
        assert(M % MMA_M == 0);
        assert(N % MMA_N == 0);
        assert(K % MMA_K == 0);

        dim3 gridDim((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M);
        dim3 blockDim(BLOCK_SIZE);
        int shmemBytes = (MMA_M * MMA_K + MMA_K * MMA_N) * sizeof(uint8_t);

        // Launch kernel
        // Grid: ((N + MMA_N - 1/ MMA_N), (M + MMA_M - 1)/ MMA_M, 1)
        // Block: (256, 1, 1)
        void *args[] = {&A_d,
                        &B_d,
                        &C_d,
                        (void*)&M,
                        (void*)&N,
                        (void*)&K};

        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaTensorMatMulM16N8k16Int8,
                                    gridDim,
                                    blockDim,
                                    args,
                                    shmemBytes,
                                    stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        cpuMatMulReference(A_h, B_h, cpuC_h, M, N, K);

        //Copying the result back to the host
        CUDA_CHECK(cudaMemcpyAsync(gpuC_h, C_d, M * N * sizeof(int), cudaMemcpyDeviceToHost, stream));

        //Validate the result
        for(int t = 0; t < M*N; ++t) {
            assert(gpuC_h[t] == cpuC_h[t]);
        }

        //Free up resources
        CUDA_CHECK(cudaFreeAsync(A_d, stream));
        CUDA_CHECK(cudaFreeAsync(B_d, stream));
        CUDA_CHECK(cudaFreeAsync(C_d, stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
        free(A_h);
        free(B_h);
        free(cpuC_h);
        free(gpuC_h);
    }
}

void benchmark() {
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const int BLOCK_SIZE = 256;
    const int WARMUP = 3;
    const int ITERS = 100;

    std::mt19937 randEngine(123);
    std::uniform_real_distribution<float> randDist(1.0f, 255.0f);

    uint8_t* A_h = (uint8_t*)malloc(M * K * sizeof(uint8_t));
    uint8_t* B_h = (uint8_t*)malloc(K * N * sizeof(uint8_t));
    int* C_h = (int*)malloc(M * N * sizeof(int));

    for (int i = 0; i < M * K; i++) A_h[i] = uint8_t((uint32_t)randDist(randEngine));
    for (int i = 0; i < K * N; i++) B_h[i] = uint8_t((uint32_t)randDist(randEngine));

    uint8_t* A_d;
    uint8_t* B_d;
    int* C_d;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocAsync(&A_d, M * K * sizeof(uint8_t), stream));
    CUDA_CHECK(cudaMallocAsync(&B_d, K * N * sizeof(uint8_t), stream));
    CUDA_CHECK(cudaMallocAsync(&C_d, M * N * sizeof(int), stream));

    CUDA_CHECK(cudaMemcpyAsync(A_d, A_h, M * K * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(B_d, B_h, K * N * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));

    dim3 gridDim((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M);
    dim3 blockDim(BLOCK_SIZE);
    int shmemBytes = (MMA_M * MMA_K + MMA_K * MMA_N) * sizeof(uint8_t);

    void *args[] = {&A_d, &B_d, &C_d, (void*)&M, (void*)&N, (void*)&K};

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        CUDA_CHECK(cudaMemsetAsync(C_d, 0, M * N * sizeof(int), stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaTensorMatMulM16N8k16Int8,
                                    gridDim, blockDim, args, shmemBytes, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed iterations
    nvtxRangePushA("bench_region");
    for (int i = 0; i < ITERS; i++) {
        CUDA_CHECK(cudaMemsetAsync(C_d, 0, M * N * sizeof(int), stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaTensorMatMulM16N8k16Int8,
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
    free(C_h);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}