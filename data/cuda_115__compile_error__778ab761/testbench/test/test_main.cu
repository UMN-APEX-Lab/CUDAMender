#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "odd_even_separation.h"
#include "cuda_helpers.h"

constexpr int NUM_BLOCKS_PER_GRID = 256;
constexpr int NUM_ELEMENTS = 10;
constexpr int NUM_OUTPUT_ELEMENTS = (NUM_ELEMENTS & 1) ? (NUM_ELEMENTS / 2 + 1) : (NUM_ELEMENTS / 2);

int launch() {
    // Ensure problem size exceeds single-pass coverage
    // Grid: NUM_BLOCKS_PER_GRID blocks × NUM_THREADS_PER_BLOCK threads
    int singlePassCoverage = NUM_BLOCKS_PER_GRID * NUM_THREADS_PER_BLOCK;
    int NUM_ELEMENTS = singlePassCoverage * 2;
    int NUM_OUTPUT_ELEMENTS = NUM_ELEMENTS / 2;

    // Use vectors instead of stack arrays since NUM_ELEMENTS is now large
    std::vector<int> input_h(NUM_ELEMENTS);
    std::vector<int> oddData_h(NUM_OUTPUT_ELEMENTS);
    std::vector<int> evenData_h(NUM_OUTPUT_ELEMENTS);
    std::vector<int> oddDataExpected_h(NUM_OUTPUT_ELEMENTS);
    std::vector<int> evenDataExpected_h(NUM_OUTPUT_ELEMENTS);

    int *input_d;
    int *oddData_d;
    int *evenData_d;

    cudaStream_t stream;

    // Allocating resources.
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocAsync(&input_d, sizeof(int) * NUM_ELEMENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&oddData_d, sizeof(int) * NUM_OUTPUT_ELEMENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&evenData_d, sizeof(int) * NUM_OUTPUT_ELEMENTS, stream));

    dim3 gridDim(NUM_BLOCKS_PER_GRID, 1, 1);
    dim3 blockDim(NUM_THREADS_PER_BLOCK, 1, 1);
    void *args[4] = { &input_d, &oddData_d, &evenData_d, (void*)&NUM_ELEMENTS };
    const int numTests = 7;
    srand(1);

    for(int test = 0; test < numTests; test++) {
        for(int i = 0; i < NUM_ELEMENTS; i++) {
            input_h[i] = rand();

            if(i & 1) {
                oddDataExpected_h[i / 2] = input_h[i];
            } else {
                evenDataExpected_h[i / 2] = input_h[i];
            }
        }

        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h.data(),
                   sizeof(int) * NUM_ELEMENTS, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_separateOddEven, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(oddData_h.data(), oddData_d,
                   sizeof(int) * NUM_OUTPUT_ELEMENTS, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(evenData_h.data(), evenData_d,
                   sizeof(int) * NUM_OUTPUT_ELEMENTS, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for(int i = 0; i < NUM_OUTPUT_ELEMENTS; i++) {
            // Number of odd-indexed elements is 1 less than number of even-indexed elements if input size is odd.
            if((NUM_ELEMENTS & 1) && (i < NUM_OUTPUT_ELEMENTS - 1) || !(NUM_ELEMENTS & 1)){
                assert(oddDataExpected_h[i] == oddData_h[i]);
            }

            assert(evenDataExpected_h[i] == evenData_h[i]);
        }
    }

    // Releasing resources.
    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(oddData_d, stream));
    CUDA_CHECK(cudaFreeAsync(evenData_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}

int benchmark() {
    constexpr int BENCH_ELEMENTS = 64 * 1024 * 1024;
    constexpr int BENCH_OUTPUT = BENCH_ELEMENTS / 2;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 2000;

    int *input_d, *oddData_d, *evenData_d;
    cudaStream_t stream;

    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocAsync(&input_d, sizeof(int) * BENCH_ELEMENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&oddData_d, sizeof(int) * BENCH_OUTPUT, stream));
    CUDA_CHECK(cudaMallocAsync(&evenData_d, sizeof(int) * BENCH_OUTPUT, stream));
    CUDA_CHECK(cudaMemsetAsync(input_d, 1, sizeof(int) * BENCH_ELEMENTS, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    dim3 grid(NUM_BLOCKS_PER_GRID, 1, 1);
    dim3 block(NUM_THREADS_PER_BLOCK, 1, 1);
    int numElem = BENCH_ELEMENTS;
    void *args[4] = { &input_d, &oddData_d, &evenData_d, &numElem };

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_separateOddEven, grid, block, args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_separateOddEven, grid, block, args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(oddData_d, stream));
    CUDA_CHECK(cudaFreeAsync(evenData_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        return benchmark();
    }
    launch();
}