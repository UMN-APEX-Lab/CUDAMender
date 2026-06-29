#include "mma_kernel.h"
#include <cstdio>
#include <cassert>
#include <random>
#include <cmath>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

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
// Inputs => ( Column Major -> A, Row Major -> B)
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
                float a_val = static_cast<float>(A[k*M + i]);
                float b_val = static_cast<float>(B[k*N + j]);
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

    const int TEST_CASES_DIMS[TEST_CASE_COUNT][PROBLEM_DIMS] = {{16,8,8}, {512,512,512}, {32,16,32}, {256, 256, 256}, {64, 64, 64} , {64, 32, 32}, {128, 128, 128}};

    //Tolerance for validation, set to 1% due to nature of half precision operations
    const float TOLERANCE  = 0.01;
    const int BLOCK_SIZE = 256;


    //Set up random number generation
    std::mt19937 randEngine(42); // Fixed seed for reproducibility

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
        for (int c = 0; c < K ; ++c) {
            for (int r = 0; r < M; ++r) {
                float val = randDist(randEngine);
                A_h[c * M + r] = __nv_bfloat16(val); // Filling A Matrix in Column Major Way
            }
       }

        for (int r = 0; r < K; ++r) {
            for (int c = 0; c < N; ++c) {
                float  val = randDist(randEngine);
                B_h[r * N + c] = __nv_bfloat16(val);
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

        // dim3 gridDim((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M);
        dim3 gridDim((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M);
        dim3 blockDim(BLOCK_SIZE);  // one warp per block
        int shmemBytes = (MMA_M * MMA_K + MMA_K * MMA_N) * sizeof(__nv_bfloat16);

        // Launch kernel
        // Grid: ((N + MMA_N - 1/ MMA_N), (M + MMA_M - 1)/ MMA_M, 1)
        // Block: (256, 1, 1)
        void *args[] = {&A_d,
                        &B_d,
                        &C_d,
                        (void*)&M,
                        (void*)&N,
                        (void*)&K};

        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaM16N8K8AcolBrow,
                                    gridDim,
                                    blockDim,
                                    args,
                                    shmemBytes,
                                    stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        cpuMatMulReference(A_h, B_h, cpuC_h, M, N, K);

        //Copying the result back to the host
        CUDA_CHECK(cudaMemcpyAsync(gpuC_h, C_d, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream));

        //Validate the result, with in 1% tolerance
        for(int t = 0; t < M*N; ++t) {
            assert(std::fabs((gpuC_h[t] - cpuC_h[t]) / std::fabs(cpuC_h[t])) <= TOLERANCE);
        }

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

void benchmark() {
    const int M = 512;
    const int N = 512;
    const int K = 512;
    const int BLOCK_SIZE = 256;
    const int WARMUP_ITERS = 3;
    const int TIMED_ITERS = 100;

    std::mt19937 randEngine(123);
    std::uniform_real_distribution<float> randDist(1.0f, 100.0f);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    __nv_bfloat16* A_h = (__nv_bfloat16*)malloc(M * K * sizeof(__nv_bfloat16));
    __nv_bfloat16* B_h = (__nv_bfloat16*)malloc(K * N * sizeof(__nv_bfloat16));

    for (int c = 0; c < K; ++c)
        for (int r = 0; r < M; ++r)
            A_h[c * M + r] = __nv_bfloat16(randDist(randEngine));

    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c)
            B_h[r * N + c] = __nv_bfloat16(randDist(randEngine));

    __nv_bfloat16 *A_d, *B_d;
    float *C_d;
    CUDA_CHECK(cudaMallocAsync(&A_d, M * K * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&B_d, K * N * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&C_d, M * N * sizeof(float), stream));

    CUDA_CHECK(cudaMemcpyAsync(A_d, A_h, M * K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(B_d, B_h, K * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));

    dim3 grid((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M);
    dim3 block(BLOCK_SIZE);
    int shmemBytes = (MMA_M * MMA_K + MMA_K * MMA_N) * sizeof(__nv_bfloat16);

    void *args[] = {&A_d, &B_d, &C_d, (void*)&M, (void*)&N, (void*)&K};

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaMemsetAsync(C_d, 0, M * N * sizeof(float), stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaM16N8K8AcolBrow,
                                    grid, block, args, shmemBytes, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaMemsetAsync(C_d, 0, M * N * sizeof(float), stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaM16N8K8AcolBrow,
                                    grid, block, args, shmemBytes, stream));
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