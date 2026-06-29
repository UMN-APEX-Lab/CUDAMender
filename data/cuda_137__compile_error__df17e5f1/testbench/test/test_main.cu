#include "binary_search.h"
#include "cuda_check.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <cooperative_groups.h>
#include <nvtx3/nvToolsExt.h>

namespace cg = cooperative_groups;

void launch() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Define test cases (sorted arrays) and corresponding query cases.
    std::vector<std::vector<int>> testCases = {
        {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21},
        {2, 4, 6, 8, 10},
        {10, 20, 30, 40, 50, 60, 70, 80, 90, 100},
        {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000},
        {9, 18, 27, 36, 45, 54, 63, 72, 81, 90},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    };

    std::vector<std::vector<int>> queryCases = {
        {5, 10, 15, 20, 21},
        {6, 3, 10},
        {25, 40, 100, 200},
        {700, 500, 1000, 1100},
        {45, 100, 9, 81},
        {1, 10, 5}
    };

    // Expected results for each test case.
    std::vector<std::vector<int>> expectedResults = {
        {2, -1, 7, -1, 10},
        {2, -1, 4},
        {-1, 3, 9, -1},
        {6, 4, 9, -1},
        {4, -1, 0, 8},
        {0, 9, 4}
    };

    // Process each test case.
    for (size_t testCaseIndex = 0; testCaseIndex < testCases.size(); testCaseIndex++) {
        int arraySize = testCases[testCaseIndex].size();
        int querySize = queryCases[testCaseIndex].size();
        if (arraySize == 0 || querySize == 0) continue; // Skip empty test cases

        // Allocate device pointers.
        int *arr_d, *queries_d, *results_d;

        // Allocate device memory.
        CUDA_CHECK(cudaMallocAsync(&arr_d, arraySize * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&queries_d, querySize * sizeof(int), stream));
        CUDA_CHECK(cudaMallocAsync(&results_d, querySize * sizeof(int), stream));

        // Copy input data.
        CUDA_CHECK(cudaMemcpyAsync(arr_d, testCases[testCaseIndex].data(), arraySize * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(queries_d, queryCases[testCaseIndex].data(), querySize * sizeof(int), cudaMemcpyHostToDevice, stream));

        // --- Begin occupancy and grid dimension setup for this test case ---
        cudaDeviceProp deviceProps;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, 0));
        cudaDeviceProp cooperativeProps;
        CUDA_CHECK(cudaGetDeviceProperties(&cooperativeProps, 0));
        assert(cooperativeProps.cooperativeLaunch && "Error: This device does not support cooperative kernel launches.");

        // Calculate the total number of Streaming Multiprocessors (SMs).
        int totalSMs = deviceProps.multiProcessorCount;

        // Determine the maximum number of blocks per SM that can be active for the binary search kernel.
        int maxBlocksPerSM;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, k_binarySearch, THREADS_PER_BLOCK, 0));

        // Calculate the maximum number of cooperative blocks allowed on the device.
        int maxCoopBlocks = maxBlocksPerSM * totalSMs;

        // Compute the number of blocks required based on the current query dataset.
        int numBlocks = (querySize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        numBlocks = std::min(numBlocks, maxCoopBlocks); // Clamp block count to cooperative launch limits.
        numBlocks = std::max(numBlocks, 1); // Ensure at least one block is launched.

        // Define grid and block dimensions.
        dim3 gridSize(numBlocks, 1, 1);
        dim3 blockSize(THREADS_PER_BLOCK, 1, 1);

        // Verify that grid dimensions do not exceed device limits.
        if (gridSize.x > deviceProps.maxGridSize[0] || gridSize.y > deviceProps.maxGridSize[1]) {
            assert(false && "Grid size exceeds device limits!");
        }

        // Prepare kernel arguments.
        void* kernelArgs[] = {
            (void*)&arr_d,
            (void*)&queries_d,
            (void*)&results_d,
            (void*)&arraySize,
            (void*)&querySize
        };

        // Launch the cooperative kernel.
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_binarySearch, gridSize, blockSize, kernelArgs, 0, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy results back.
        std::vector<int> output_h(querySize);
        CUDA_CHECK(cudaMemcpyAsync(output_h.data(), results_d, querySize * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Verify the result for this test case.
        for (int index = 0; index < querySize; index++) {
            assert(output_h[index] == expectedResults[testCaseIndex][index]);
        }

        // Free device memory.
        CUDA_CHECK(cudaFreeAsync(arr_d, stream));
        CUDA_CHECK(cudaFreeAsync(queries_d, stream));
        CUDA_CHECK(cudaFreeAsync(results_d, stream));
    }

    // Destroy CUDA stream.
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char* argv[]) {
    bool perf = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--perf") == 0) perf = true;
    }

    if (perf) {
        const int arrSize = 4 * 1024 * 1024;
        const int querySize = 2 * 1024 * 1024;
        const int warmup_iters = 3;
        const int timed_iters = 100;

        std::vector<int> arr_h(arrSize);
        for (int i = 0; i < arrSize; i++) arr_h[i] = i * 2;

        std::vector<int> queries_h(querySize);
        for (int i = 0; i < querySize; i++) queries_h[i] = (i * 3) % (arrSize * 2);

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        int *arr_d, *queries_d, *results_d;
        CUDA_CHECK(cudaMalloc(&arr_d, arrSize * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&queries_d, querySize * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&results_d, querySize * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(arr_d, arr_h.data(), arrSize * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(queries_d, queries_h.data(), querySize * sizeof(int), cudaMemcpyHostToDevice));

        cudaDeviceProp deviceProps;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProps, 0));
        int totalSMs = deviceProps.multiProcessorCount;
        int maxBlocksPerSM;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, k_binarySearch, THREADS_PER_BLOCK, 0));
        int maxCoopBlocks = maxBlocksPerSM * totalSMs;
        int numBlocks = (querySize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        numBlocks = std::min(numBlocks, maxCoopBlocks);
        numBlocks = std::max(numBlocks, 1);

        dim3 gridSize(numBlocks, 1, 1);
        dim3 blockSize(THREADS_PER_BLOCK, 1, 1);

        void* kernelArgs[] = {
            (void*)&arr_d,
            (void*)&queries_d,
            (void*)&results_d,
            (void*)&arrSize,
            (void*)&querySize
        };

        for (int i = 0; i < warmup_iters; i++) {
            CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_binarySearch, gridSize, blockSize, kernelArgs, 0, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        nvtxRangePushA("bench_region");
        for (int i = 0; i < timed_iters; i++) {
            CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_binarySearch, gridSize, blockSize, kernelArgs, 0, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        nvtxRangePop();

        CUDA_CHECK(cudaFree(arr_d));
        CUDA_CHECK(cudaFree(queries_d));
        CUDA_CHECK(cudaFree(results_d));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        launch();
    }
    return 0;
}