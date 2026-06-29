#include "reduce.h"
#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <nvtx3/nvToolsExt.h>

#define cudaCheckErrors(msg)                                                                 \
    do                                                                                       \
    {                                                                                        \
        cudaError_t __err = cudaGetLastError();                                              \
        if (__err != cudaSuccess)                                                            \
        {                                                                                    \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)", msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__);                                                     \
            fprintf(stderr, "*** FAILED - ABORTING");                                        \
            exit(1);                                                                         \
        }                                                                                    \
    }                                                                                        \
    while (0)

bool validate(float *h_A, float *h_sum, size_t n)
{
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < n; i++)
    {
        if (h_A[i] > max_val)
        {
            max_val = h_A[i];
        }
    }
    return fabs(*h_sum - max_val) < 1e-5;
}

int main(int argc, char *argv[]) {

    auto launch = []() -> void {

        float *h_A, *h_sum, *d_A, *d_sums;
        const int blocks = 640;
        const size_t N   = 8ULL * 1024ULL * 1024ULL;
        h_A              = new float[N];
        h_sum            = new float;

        // Initialize random seed
        srand(42);

        // Initialize the array with random numbers
        float max_val = -FLT_MAX;
        for (size_t i = 0; i < N; i++)
        {
            h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            if (h_A[i] > max_val)
            {
                max_val = h_A[i];
            }
        }

        cudaMalloc(&d_A, N * sizeof(float));
        cudaMalloc(&d_sums, blocks * sizeof(float));
        cudaCheckErrors("cudaMalloc failure");

        cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy H2D failure");

        reduce<<<blocks, BLOCK_SIZE>>>(d_A, d_sums, N);
        cudaCheckErrors("reduction kernel launch failure");

        reduce<<<1, BLOCK_SIZE>>>(d_sums, d_A, blocks);
        cudaCheckErrors("reduction kernel launch failure");

        cudaMemcpy(h_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy D2H failure");

        assert(validate(h_A, h_sum, N));


        // Put the maximum value at the end of the array to check if the program correctly handles arrays longer than the no of threads
        max_val = 1.1;
        h_A[N - 1] = max_val;

        cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaCheckErrors("cudaMemcpy H2D failure");

        reduce<<<blocks, BLOCK_SIZE>>>(d_A, d_sums, N);
        cudaCheckErrors("reduction kernel launch failure");

        reduce<<<1, BLOCK_SIZE>>>(d_sums, d_A, blocks);
        cudaCheckErrors("reduction kernel launch failure");

        cudaMemcpy(h_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy D2H failure");

        cudaMemcpy(h_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy D2H failure");

        assert(validate(h_A, h_sum, N));

        delete[] h_A;
        delete h_sum;
        cudaFree(d_A);
        cudaFree(d_sums);
    };

    launch();

    if (argc > 1 && strcmp(argv[1], "--perf") == 0)
    {
        const size_t PERF_N      = 128ULL * 1024ULL * 1024ULL;
        const int    perf_blocks = 2560;
        const int    warmup      = 3;
        const int    timed       = 100;

        float *h_perf = new float[PERF_N];
        srand(123);
        for (size_t i = 0; i < PERF_N; i++)
            h_perf[i] = static_cast<float>(rand()) / RAND_MAX;

        float *d_perf, *d_perf_sums;
        cudaMalloc(&d_perf, PERF_N * sizeof(float));
        cudaMalloc(&d_perf_sums, perf_blocks * sizeof(float));
        cudaMemcpy(d_perf, h_perf, PERF_N * sizeof(float), cudaMemcpyHostToDevice);

        for (int i = 0; i < warmup; i++)
        {
            reduce<<<perf_blocks, BLOCK_SIZE>>>(d_perf, d_perf_sums, PERF_N);
            reduce<<<1, BLOCK_SIZE>>>(d_perf_sums, d_perf, perf_blocks);
        }
        cudaDeviceSynchronize();

        nvtxRangePushA("bench_region");
        for (int i = 0; i < timed; i++)
        {
            reduce<<<perf_blocks, BLOCK_SIZE>>>(d_perf, d_perf_sums, PERF_N);
            reduce<<<1, BLOCK_SIZE>>>(d_perf_sums, d_perf, perf_blocks);
        }
        cudaDeviceSynchronize();
        nvtxRangePop();

        delete[] h_perf;
        cudaFree(d_perf);
        cudaFree(d_perf_sums);
    }

}