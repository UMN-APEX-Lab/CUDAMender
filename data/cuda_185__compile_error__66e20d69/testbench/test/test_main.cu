#undef NDEBUG
#include "log2_kernel.h"
#include <assert.h>
#include <stdio.h>
#include <random>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK(call) {                                     \
    cudaError_t error = call;                                  \
    if(error != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

// Maximum number of elements for input and output.
constexpr int MAX_DATA_ELEMENTS = 1000000;
constexpr int NUM_TESTS = 7;
constexpr float ERROR_TOLERANCE = 0.008f;
constexpr int DETERMINISTIC_RANDOM_SEED = 42;
// The number of memory banks in shared memory since the Fermi architecture.
constexpr int NUM_MEMORY_BANKS = 32;

void launch() {
    // The ideal number of duplicates per lookup table element to ensure the same data in all available memory banks.
    int lookupTableDuplication = NUM_MEMORY_BANKS;
    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));
    int numSM = deviceProperties.multiProcessorCount;
    // The shared memory allocation adjustment to fit inside the available size.
    size_t dynamicSMemSize = sizeof(float) * lookupTableDuplication * LOOKUP_TABLE_ELEMENTS;
    while(dynamicSMemSize > deviceProperties.sharedMemPerBlock) {
        lookupTableDuplication--;
        dynamicSMemSize = sizeof(float) * lookupTableDuplication * LOOKUP_TABLE_ELEMENTS;
    }
    int maxBlocksPerSM;
    int maxBlockSize;
    int minGridSize;
    // Querying the largest possible block size.
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, (void*)k_calculateLog2, dynamicSMemSize));
    // Queryint the maximum number of blocks per SM.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, (void*)k_calculateLog2, maxBlockSize, dynamicSMemSize));
    int maxBlocks = maxBlocksPerSM * numSM;
    // Allocating host memory.
    float * lookupTable_h = new float[LOOKUP_TABLE_ELEMENTS];
    float * data_h = new float[MAX_DATA_ELEMENTS];
    float * testData_h = new float[MAX_DATA_ELEMENTS * NUM_TESTS];
    int * testNumElements_h = new int[NUM_TESTS];
    // Allocating device memory.
    float * lookupTable_d;
    float * data_d;
    CUDA_CHECK(cudaMallocAsync(&lookupTable_d, sizeof(float) * LOOKUP_TABLE_ELEMENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&data_d, sizeof(float) * MAX_DATA_ELEMENTS, stream));
    
    // Initializing the lookup table data.
    for(int i = 0; i < LOOKUP_TABLE_ELEMENTS; i++) {
        lookupTable_h[i] = log2f(1.0f + i / (float) LOOKUP_TABLE_ELEMENTS);
    }
    CUDA_CHECK(cudaMemcpyAsync(lookupTable_d, lookupTable_h, sizeof(float) * LOOKUP_TABLE_ELEMENTS, cudaMemcpyHostToDevice, stream));
    
    // Test 1: 1.0f / (2 ^ (i + 1))
    int testIndex = 0;
    {
        int numElements = 10;
        testNumElements_h[testIndex] = numElements;
        for(int i = 0; i < numElements; i++) {
            testData_h[i + testIndex * MAX_DATA_ELEMENTS] = 1.0f / pow(2.0f, i + 1);
        }
        testIndex++;
    }
    // Test 2: 1.0f / (i + 1)
    {
        int numElements = 10;
        testNumElements_h[testIndex] = numElements;
        for(int i = 0; i < numElements; i++) {
            testData_h[i + testIndex * MAX_DATA_ELEMENTS] = 1.0f / (float)(i + 1);
        }
        testIndex++;
    }
    // Test 3: Random normalized values.
    {
        std::mt19937 generator(DETERMINISTIC_RANDOM_SEED);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        int numElements = MAX_DATA_ELEMENTS;
        testNumElements_h[testIndex] = numElements;
        for(int i = 0; i < numElements; i++) {
            testData_h[i + testIndex * MAX_DATA_ELEMENTS] = distribution(generator);
        }
        testIndex++;
    }
    // Test 4: Elements with values close to 0.0f.
    {
        int numElements = MAX_DATA_ELEMENTS;
        testNumElements_h[testIndex] = numElements;
        for(int i = 0; i < numElements; i++) {
            testData_h[i + testIndex * MAX_DATA_ELEMENTS] = 0.00001f +  0.0001f * i / (float)numElements;
        }
        testIndex++;
    }
    // Test 5: Elements with values close to 1.0f.
    {
        int numElements = MAX_DATA_ELEMENTS;
        testNumElements_h[testIndex] = numElements;
        for(int i = 0; i < numElements; i++) {
            testData_h[i + testIndex * MAX_DATA_ELEMENTS] = 1.0f - 0.0001f * i / (float)numElements;
        }
        testIndex++;
    }
    // Test 6: Logistic map iterations for r = 3.7f and x = 0.14f
    {
        int numElements = MAX_DATA_ELEMENTS;
        testNumElements_h[testIndex] = numElements;
        float x = 0.14f;
        float r = 3.7f;
        for(int i = 0; i < numElements; i++) {
            testData_h[i + testIndex * MAX_DATA_ELEMENTS] = x;
            x = r * (1.0f - x) * x;
        }
        testIndex++;
    }
    // Test 7: Sine map iterations for x = 0.5f and r = 0.99f.
    {
        int numElements = MAX_DATA_ELEMENTS;
        testNumElements_h[testIndex] = numElements;
        float x = 0.5f;
        float r = 0.99f;
        for(int i = 0; i < numElements; i++) {
            testData_h[i + testIndex * MAX_DATA_ELEMENTS] = x;
            x = r * sin(x * acosf(-1.0f));
        }
        testIndex++;
    }
    // Iterating the tests.
    for(int test = 0; test < testIndex; test++)
    {
        int numElements = testNumElements_h[test];
        for(int i = 0; i < numElements; i++) {
            data_h[i] = testData_h[i + test * MAX_DATA_ELEMENTS];
        }
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, sizeof(float) * numElements, cudaMemcpyHostToDevice, stream));
        void * args[4] = { &numElements, &data_d, &lookupTable_d, &lookupTableDuplication };
        int requiredBlocks = (numElements + maxBlockSize - 1) / maxBlockSize;
        int usedBlocks = requiredBlocks < maxBlocks ? requiredBlocks : maxBlocks;
        // Grid: (usedBlocks, 1, 1)
        // Block: (maxBlockSize, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLog2, dim3(usedBlocks, 1, 1), dim3(maxBlockSize, 1, 1), args, dynamicSMemSize, stream));
        CUDA_CHECK(cudaMemcpyAsync(data_h, data_d, sizeof(float) * numElements, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < numElements; i++) {
           float expectedValue = log2f(testData_h[i + test * MAX_DATA_ELEMENTS]);
           assert(fabsf(expectedValue - data_h[i]) < ERROR_TOLERANCE);
        }
    }
    // Freeing device memory.
    CUDA_CHECK(cudaFreeAsync(lookupTable_d, stream));
    CUDA_CHECK(cudaFreeAsync(data_d, stream));
    // Freeing host memory.
    delete [] lookupTable_h;
    delete [] data_h;
    delete [] testData_h;
    delete [] testNumElements_h;
    // Freeing the stream.
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    constexpr int BENCH_ELEMENTS = 10000000;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 100;

    int lookupTableDuplication = NUM_MEMORY_BANKS;
    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));
    int numSM = deviceProperties.multiProcessorCount;
    size_t dynamicSMemSize = sizeof(float) * lookupTableDuplication * LOOKUP_TABLE_ELEMENTS;
    while(dynamicSMemSize > deviceProperties.sharedMemPerBlock) {
        lookupTableDuplication--;
        dynamicSMemSize = sizeof(float) * lookupTableDuplication * LOOKUP_TABLE_ELEMENTS;
    }
    int maxBlocksPerSM;
    int maxBlockSize;
    int minGridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, (void*)k_calculateLog2, dynamicSMemSize));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, (void*)k_calculateLog2, maxBlockSize, dynamicSMemSize));
    int maxBlocks = maxBlocksPerSM * numSM;

    float * lookupTable_h = new float[LOOKUP_TABLE_ELEMENTS];
    float * data_h = new float[BENCH_ELEMENTS];
    float * lookupTable_d;
    float * data_d;
    CUDA_CHECK(cudaMallocAsync(&lookupTable_d, sizeof(float) * LOOKUP_TABLE_ELEMENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&data_d, sizeof(float) * BENCH_ELEMENTS, stream));

    for(int i = 0; i < LOOKUP_TABLE_ELEMENTS; i++) {
        lookupTable_h[i] = log2f(1.0f + i / (float) LOOKUP_TABLE_ELEMENTS);
    }
    CUDA_CHECK(cudaMemcpyAsync(lookupTable_d, lookupTable_h, sizeof(float) * LOOKUP_TABLE_ELEMENTS, cudaMemcpyHostToDevice, stream));

    std::mt19937 generator(123);
    std::uniform_real_distribution<float> distribution(0.001f, 0.999f);
    for(int i = 0; i < BENCH_ELEMENTS; i++) {
        data_h[i] = distribution(generator);
    }

    int requiredBlocks = (BENCH_ELEMENTS + maxBlockSize - 1) / maxBlockSize;
    int usedBlocks = requiredBlocks < maxBlocks ? requiredBlocks : maxBlocks;
    void * args[4] = { nullptr, &data_d, &lookupTable_d, &lookupTableDuplication };
    int numElements = BENCH_ELEMENTS;
    args[0] = &numElements;

    for(int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, sizeof(float) * BENCH_ELEMENTS, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLog2, dim3(usedBlocks, 1, 1), dim3(maxBlockSize, 1, 1), args, dynamicSMemSize, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for(int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, sizeof(float) * BENCH_ELEMENTS, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLog2, dim3(usedBlocks, 1, 1), dim3(maxBlockSize, 1, 1), args, dynamicSMemSize, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(lookupTable_d, stream));
    CUDA_CHECK(cudaFreeAsync(data_d, stream));
    delete [] lookupTable_h;
    delete [] data_h;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char ** argv) {
    if(argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}