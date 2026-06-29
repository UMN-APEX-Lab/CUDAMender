#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <random>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include "count_occurrences.h"

#define CUDA_CHECK(call) {                                     \
    cudaError_t error = call;                                  \
    if (error != cudaSuccess) {                                \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

// Test settings.
constexpr int MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT = 1000000;
constexpr int MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST = 1000;
constexpr int ALLOCATION_SIZE_FOR_ELEMENTS_TO_COUNT = MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT * sizeof(int);
constexpr int ALLOCATION_SIZE_FOR_ELEMENTS_TO_COMPARE_AGAINST = MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST * sizeof(int);
constexpr int NUM_TESTS = 7;

void launch() {
    constexpr int DETERMINISTIC_RANDOM_SEED = 42;
    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));
    
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Allocating host buffers.
    int * array1_h = new int[MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT];
    int * array2_h = new int[MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST];
    int * count_h = new int[MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT];
    int * testNumIntegersToCount_h = new int[NUM_TESTS];
    int * testNumIntegersToCompare_h = new int[NUM_TESTS];
    int * testIntegersToCount_h = new int[NUM_TESTS * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT];
    int * testIntegersToCompare_h = new int[NUM_TESTS * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST];
    int * testResult_h = new int[NUM_TESTS * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT];
    // Allocating device buffers.
    int * array1_d;
    int * array2_d;
    int * count_d;
    CUDA_CHECK(cudaMallocAsync(&array1_d, ALLOCATION_SIZE_FOR_ELEMENTS_TO_COUNT, stream));
    CUDA_CHECK(cudaMallocAsync(&array2_d, ALLOCATION_SIZE_FOR_ELEMENTS_TO_COMPARE_AGAINST, stream));
    CUDA_CHECK(cudaMallocAsync(&count_d, ALLOCATION_SIZE_FOR_ELEMENTS_TO_COUNT, stream));
    // Preparing the tests.
    int testIndex = 0;
    // Test 1
    {
        int counts = 300000;
        int comparisons = 70;
        testNumIntegersToCount_h[testIndex] = counts;
        testNumIntegersToCompare_h[testIndex] = comparisons;
        for (int i = 0; i < counts; i++) {
            testIntegersToCount_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = i;
        }
        for (int i = 0; i < comparisons; i++) {
            testIntegersToCompare_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST] = i % 2;
        }
        for (int i = 0; i < counts; i++) {
            testResult_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = (i < 2 ? comparisons / 2 : 0);
        }
        testIndex++;
    }
    // Test 2
    {
        int counts = 200000;
        int comparisons = 10;
        testNumIntegersToCount_h[testIndex] = counts;
        testNumIntegersToCompare_h[testIndex] = comparisons;
        for (int i = 0; i < counts; i++) {
            testIntegersToCount_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = i % comparisons;
        }
        for (int i = 0; i < comparisons; i++) {
            testIntegersToCompare_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST] = i;
        }
        for (int i = 0; i < counts; i++) {
            testResult_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = 1;
        }
        testIndex++;
    }
    // Test 3
    {
        int counts = 400000;
        int comparisons = 100;
        testNumIntegersToCount_h[testIndex] = counts;
        testNumIntegersToCompare_h[testIndex] = comparisons;
        for (int i = 0; i < counts; i++) {
            testIntegersToCount_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = i * 4;
        }
        for (int i = 0; i < comparisons; i++) {
            testIntegersToCompare_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST] = i / 4;
        }
        for (int i = 0; i < counts; i++) {
            testResult_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = i * 4 < comparisons / 4 ? 4 : 0;
        }
        testIndex++;
    }
    // Test 4
    {
        int counts = 500000;
        int comparisons = 10;
        testNumIntegersToCount_h[testIndex] = counts;
        testNumIntegersToCompare_h[testIndex] = comparisons;
        for (int i = 0; i < counts; i++) {
            testIntegersToCount_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = 1;
        }
        for (int i = 0; i < comparisons; i++) {
            testIntegersToCompare_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST] = i;
        }
        for (int i = 0; i < counts; i++) {
            testResult_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = (testIntegersToCount_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] == 1 ? 1 : 0);
        }
        testIndex++;
    }
    // Test 5
    {
        int counts = 700000;
        int comparisons = 70;
        testNumIntegersToCount_h[testIndex] = counts;
        testNumIntegersToCompare_h[testIndex] = comparisons;
        for (int i = 0; i < counts; i++) {
            testIntegersToCount_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = i;
        }
        for (int i = 0; i < comparisons; i++) {
            testIntegersToCompare_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST] = i * 10000;
        }
        for (int i = 0; i < counts; i++) {
            testResult_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = ((i / 10000) * 10000 == i) ? 1 : 0;
        }
        testIndex++;
    }
    // Test 6
    {
        int counts = MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT;
        int comparisons = 1;
        testNumIntegersToCount_h[testIndex] = counts;
        testNumIntegersToCompare_h[testIndex] = comparisons;
        for (int i = 0; i < counts; i++) {
            testIntegersToCount_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = 0;
        }
        for (int i = 0; i < comparisons; i++) {
            testIntegersToCompare_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST] = 0;
        }
        for (int i = 0; i < counts; i++) {
            testResult_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = 1;
        }
        testIndex++;
    }
    // Test 7
    {
        std::mt19937 generator(DETERMINISTIC_RANDOM_SEED);
        std::uniform_int_distribution<int> distribution(-1000, 1000);
        
        int counts = MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT;
        int comparisons = MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST;
        testNumIntegersToCount_h[testIndex] = counts;
        testNumIntegersToCompare_h[testIndex] = comparisons;
        for (int i = 0; i < counts; i++) {
            testResult_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = 0;
        }
        
        for (int i = 0; i < comparisons; i++) {
            testIntegersToCompare_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST] = distribution(generator);
        }
        for (int i = 0; i < counts; i++) {
            testIntegersToCount_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] = distribution(generator);
            for (int j = 0; j < comparisons; j++) {
                int value1 = testIntegersToCount_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT];
                int value2 = testIntegersToCompare_h[j + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST];
                testResult_h[i + testIndex * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT] += (value1 == value2);
            }
        }
        
        testIndex++;
    }
    // Processing the test inputs.
    for (int testId = 0; testId < NUM_TESTS; testId++)
    {
        int len1 = testNumIntegersToCount_h[testId];
        int len2 = testNumIntegersToCompare_h[testId];
        for (int i = 0; i < len1; i++) {
            array1_h[i] = testIntegersToCount_h[i + testId * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT];
        }
        for (int i = 0; i < len2; i++) {
            array2_h[i] = testIntegersToCompare_h[i + testId * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COMPARE_AGAINST];
        }
        CUDA_CHECK(cudaMemcpyAsync(array1_d, array1_h, len1 * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(array2_d, array2_h, len2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        void * args[5] = { (void*)&array1_d, (void*)&array2_d, (void*)&len1, (void*)&len2, (void*)&count_d };
        
        int minGridSize;
        int blockSize;
        
        // Computing the minimum grid size and maximum potential block size that allows high occupancy.
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, (void*)k_countOccurrences, [](int blockSize) { 
            return blockSize * sizeof(int);
        }));
        size_t sizeOfRequiredSharedMemory = blockSize * sizeof(int);
        // Using the minimum grid size that can ensure maximum occupancy.
        // Grid: (minGridSize, 1, 1)
        // Block: (blockSize, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_countOccurrences, dim3(minGridSize, 1, 1), dim3(blockSize, 1, 1), args, sizeOfRequiredSharedMemory, stream));
        CUDA_CHECK(cudaMemcpyAsync(count_h, count_d, len1 * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < len1; i++) {
            assert(count_h[i] == testResult_h[i + testId * MAXIMUM_NUMBER_OF_ELEMENTS_TO_COUNT]);
        }
    }
    
    CUDA_CHECK(cudaFreeAsync(array1_d, stream));
    CUDA_CHECK(cudaFreeAsync(array2_d, stream));
    CUDA_CHECK(cudaFreeAsync(count_d, stream));
    delete [] array1_h;
    delete [] array2_h;
    delete [] count_h;
    delete [] testNumIntegersToCount_h;
    delete [] testNumIntegersToCompare_h;
    delete [] testIntegersToCount_h;
    delete [] testIntegersToCompare_h;
    delete [] testResult_h;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    constexpr int BENCH_LEN1 = 1000000;
    constexpr int BENCH_LEN2 = 1000;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 5000;

    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int *array1_h = new int[BENCH_LEN1];
    int *array2_h = new int[BENCH_LEN2];

    std::mt19937 gen(123);
    std::uniform_int_distribution<int> dist(-500, 500);
    for (int i = 0; i < BENCH_LEN1; i++) array1_h[i] = dist(gen);
    for (int i = 0; i < BENCH_LEN2; i++) array2_h[i] = dist(gen);

    int *array1_d, *array2_d, *count_d;
    CUDA_CHECK(cudaMalloc(&array1_d, BENCH_LEN1 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&array2_d, BENCH_LEN2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&count_d, BENCH_LEN1 * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(array1_d, array1_h, BENCH_LEN1 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(array2_d, array2_h, BENCH_LEN2 * sizeof(int), cudaMemcpyHostToDevice));

    int len1 = BENCH_LEN1;
    int len2 = BENCH_LEN2;
    void *args[5] = { (void*)&array1_d, (void*)&array2_d, (void*)&len1, (void*)&len2, (void*)&count_d };

    int minGridSize, blockSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize, &blockSize, (void*)k_countOccurrences, [](int blockSize) {
        return blockSize * sizeof(int);
    }));
    size_t sharedMem = blockSize * sizeof(int);

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_countOccurrences, dim3(minGridSize, 1, 1), dim3(blockSize, 1, 1), args, sharedMem, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_countOccurrences, dim3(minGridSize, 1, 1), dim3(blockSize, 1, 1), args, sharedMem, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFree(array1_d));
    CUDA_CHECK(cudaFree(array2_d));
    CUDA_CHECK(cudaFree(count_d));
    delete[] array1_h;
    delete[] array2_h;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}
