#include "prefix_sum.h"
#include "cuda_utils.h"
#include <vector>
#include <cassert>
#include <algorithm>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

using namespace std;

struct TestCase {
    vector<int> inputData;    // Input array
    vector<int> expectedResult; // Expected inclusive prefix-sum result
};

void launch() {
    constexpr int THREADS_PER_BLOCK = 256;

    // Query device properties 
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));
    
    int deviceWarpSize = deviceProperties.warpSize;
    int warpsPerBlock = (THREADS_PER_BLOCK + deviceWarpSize - 1) / deviceWarpSize;
    
    // Define comprehensive test cases
    vector<TestCase> testCases = {
        // Test Case 1: Small array of ones
        { {1, 1, 1, 1, 1}, {1, 2, 3, 4, 5} },
        // Test Case 2: Array of zeros
        { {0, 0, 0, 0}, {0, 0, 0, 0} },
        // Test Case 3: Mixed small array {3,1,4,1,5} → {3,4,8,9,14}
        { {3, 1, 4, 1, 5}, {3, 4, 8, 9, 14} },
        // Test Case 4: Single element
        { {5}, {5} },
        // Test Case 5: Increasing sequence 1..10
        { {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {1, 3, 6, 10, 15, 21, 28, 36, 45, 55} },
        // Test Case 6: Large array of 10,000 ones
        { vector<int>(10000, 1), []()->vector<int>{
              vector<int> expectedValues(10000);
              for (int i = 0; i < 10000; i++) { 
                  expectedValues[i] = i + 1; 
              }
              return expectedValues;
          }() },
        // Test Case 7: Very large array of 500,000 ones
        { vector<int>(500000, 1), []()->vector<int>{
              vector<int> expectedValues(500000);
              for (int i = 0; i < 500000; i++) { 
                  expectedValues[i] = i + 1; 
              }
              return expectedValues;
          }() }
    };

    // Find maximum element count among all test cases
    int maxElementCount = 0;
    for (const auto &testCase : testCases) {
        maxElementCount = max(maxElementCount, (int)testCase.inputData.size());
    }

    // Calculate memory requirements
    size_t maxInputBytes = (size_t)maxElementCount * sizeof(int);
    int maxBlocksNeeded = (maxElementCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t blockSumsBytes = (size_t)(maxBlocksNeeded + 1) * sizeof(int);

    // Allocate device memory buffers
    int *inputArray_d = nullptr;
    int *outputArray_d = nullptr;
    int *blockSums_d = nullptr;

    CUDA_CHECK(cudaMalloc(&inputArray_d, maxInputBytes));
    CUDA_CHECK(cudaMalloc(&outputArray_d, maxInputBytes));
    CUDA_CHECK(cudaMalloc(&blockSums_d, blockSumsBytes));

    // Create CUDA stream for asynchronous operations
    cudaStream_t computeStream;
    CUDA_CHECK(cudaStreamCreate(&computeStream));

    // Process each test case
    for (size_t testIndex = 0; testIndex < testCases.size(); testIndex++) {
        const auto &currentTest = testCases[testIndex];
        int elementCount = (int)currentTest.inputData.size();
        size_t inputBytesRequired = (size_t)elementCount * sizeof(int);

        // Copy input data to device
        CUDA_CHECK(cudaMemcpyAsync(
            inputArray_d,
            currentTest.inputData.data(),
            inputBytesRequired,
            cudaMemcpyHostToDevice,
            computeStream
        ));
        
        // Determine launch configuration
        int totalSMs, maxBlocksPerSM;
        totalSMs = deviceProperties.multiProcessorCount;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxBlocksPerSM,
            k_prefixSum,
            THREADS_PER_BLOCK,
            warpsPerBlock * sizeof(int) + sizeof(int)  // Dynamic shared memory size
        ));

        int maxActiveBlocks = maxBlocksPerSM * totalSMs;    
        // Calculate optimal grid configuration
        int blocksNeeded = (elementCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        int numBlocks = min(blocksNeeded, maxActiveBlocks);

        dim3 blockDimensions(THREADS_PER_BLOCK, 1, 1);
        dim3 gridDimensions(
            min(numBlocks, deviceProperties.maxGridSize[0]),
            1,
            1
        );
        
        // Calculate dynamic shared memory size
        size_t sharedMemoryBytes = warpsPerBlock * sizeof(int) + sizeof(int);

        // Launch cooperative kernel
        void* kernelArguments[] = {
            (void*)&inputArray_d,
            (void*)&outputArray_d,
            (void*)&blockSums_d,
            (void*)&elementCount,
            (void*)&warpsPerBlock
        };
        
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)k_prefixSum,
            gridDimensions,
            blockDimensions,
            kernelArguments,
            sharedMemoryBytes,
            computeStream
        ));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        // Copy results back to host
        vector<int> computedResult(elementCount);
        CUDA_CHECK(cudaMemcpyAsync(
            computedResult.data(),
            outputArray_d,
            inputBytesRequired,
            cudaMemcpyDeviceToHost,
            computeStream
        ));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        for (int i = 0; i < elementCount; i++) {
            assert(computedResult[i] == currentTest.expectedResult[i]);
        }
         
    }

    // Cleanup resources
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaFree(inputArray_d));
    CUDA_CHECK(cudaFree(outputArray_d));
    CUDA_CHECK(cudaFree(blockSums_d));
  
}

void benchmark() {
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int N = 16 * 1024 * 1024; // 16M elements
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 100;

    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    int deviceWarpSize = deviceProperties.warpSize;
    int warpsPerBlock = (THREADS_PER_BLOCK + deviceWarpSize - 1) / deviceWarpSize;

    size_t inputBytes = (size_t)N * sizeof(int);
    int blocksNeeded = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t blockSumsBytes = (size_t)(blocksNeeded + 1) * sizeof(int);

    int *inputArray_d = nullptr;
    int *outputArray_d = nullptr;
    int *blockSums_d = nullptr;

    CUDA_CHECK(cudaMalloc(&inputArray_d, inputBytes));
    CUDA_CHECK(cudaMalloc(&outputArray_d, inputBytes));
    CUDA_CHECK(cudaMalloc(&blockSums_d, blockSumsBytes));

    // Fill input with ones
    vector<int> hostInput(N, 1);
    CUDA_CHECK(cudaMemcpy(inputArray_d, hostInput.data(), inputBytes, cudaMemcpyHostToDevice));

    // Determine launch configuration
    int maxBlocksPerSM;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        k_prefixSum,
        THREADS_PER_BLOCK,
        warpsPerBlock * sizeof(int) + sizeof(int)
    ));

    int maxActiveBlocks = maxBlocksPerSM * deviceProperties.multiProcessorCount;
    int numBlocks = min(blocksNeeded, maxActiveBlocks);

    dim3 blockDimensions(THREADS_PER_BLOCK, 1, 1);
    dim3 gridDimensions(min(numBlocks, deviceProperties.maxGridSize[0]), 1, 1);
    size_t sharedMemoryBytes = warpsPerBlock * sizeof(int) + sizeof(int);

    int elementCount = N;
    void* kernelArguments[] = {
        (void*)&inputArray_d,
        (void*)&outputArray_d,
        (void*)&blockSums_d,
        (void*)&elementCount,
        (void*)&warpsPerBlock
    };

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)k_prefixSum, gridDimensions, blockDimensions,
            kernelArguments, sharedMemoryBytes, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed region
    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)k_prefixSum, gridDimensions, blockDimensions,
            kernelArguments, sharedMemoryBytes, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    CUDA_CHECK(cudaFree(inputArray_d));
    CUDA_CHECK(cudaFree(outputArray_d));
    CUDA_CHECK(cudaFree(blockSums_d));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}