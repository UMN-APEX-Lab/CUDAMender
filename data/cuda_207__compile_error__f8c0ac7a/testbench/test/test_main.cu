#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include "kernel.h"
#include "cuda_helpers.h"

void launch() {
    // Algorithm settings.
    constexpr uint32_t BITS_PER_BYTE = 8;
    constexpr uint32_t ELEMENTS_PER_BYTE = BITS_PER_BYTE;
    constexpr size_t BYTES_PER_INTEGER = sizeof(uint32_t);
    constexpr uint32_t ELEMENTS_PER_INTEGER = BYTES_PER_INTEGER * ELEMENTS_PER_BYTE;
    constexpr size_t PADDING = FOUR_INTEGERS_PER_THREAD;
    // Test settings.
    constexpr uint32_t NUM_TESTS = 7;
    constexpr uint32_t DETERMINISTIC_RANDOM_SEED = 100;

    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));

    int blockSize = 256;
    int maxBlocks = (deviceProperties.multiProcessorCount * deviceProperties.maxThreadsPerMultiProcessor) / blockSize;

    // Ensure MAX_NUM_ELEMENTS exceeds single-pass grid coverage,
    // requiring a grid-stride loop for correct results.
    // Each thread processes FOUR_INTEGERS_PER_THREAD integers,
    // each integer holds ELEMENTS_PER_INTEGER boolean elements.
    const size_t maxElementsPerPass = (size_t)maxBlocks * blockSize
                                    * FOUR_INTEGERS_PER_THREAD
                                    * ELEMENTS_PER_INTEGER;
    const size_t MAX_NUM_ELEMENTS = maxElementsPerPass * 2;
    const size_t MAX_NUM_INTEGERS = (MAX_NUM_ELEMENTS + ELEMENTS_PER_INTEGER - 1) / ELEMENTS_PER_INTEGER;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Allocating host memory.
    uint32_t* integers_h = new uint32_t[MAX_NUM_INTEGERS + PADDING];
    bool* elementValues_h = new bool[MAX_NUM_ELEMENTS];
    bool* testElementValues_h = new bool[NUM_TESTS * MAX_NUM_ELEMENTS];
    size_t* testNumElements_h = new size_t[NUM_TESTS];
    
    // Allocating device memory.
    uint32_t* integersIn_d;
    uint32_t* integersOut_d;
    CUDA_CHECK(cudaMallocAsync(&integersIn_d, BYTES_PER_INTEGER * (MAX_NUM_INTEGERS + PADDING), stream));
    CUDA_CHECK(cudaMallocAsync(&integersOut_d, BYTES_PER_INTEGER * (MAX_NUM_INTEGERS + PADDING), stream));
    CUDA_CHECK(cudaMemsetAsync(integersOut_d, 0, BYTES_PER_INTEGER * MAX_NUM_INTEGERS, stream));
    
    int testIndex = 0;
    // Test 1: 120 boolean elements with alternating values.
    {
        testNumElements_h[testIndex] = 120;
        // This is encoded as 2863311530, 2863311530, 2863311530, 11184810
        std::initializer_list<bool> inputs = { 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 
        };
        std::copy(inputs.begin(), inputs.end(), &testElementValues_h[testIndex * MAX_NUM_ELEMENTS]);
        testIndex++;
    }
    // Test 2: Eight boolean elements with four consecutive zero values and four consecutive one values.
    {
        testNumElements_h[testIndex] = 8;
        // This is encoded as 240, or 0b00000000000000000000000011110000
        std::initializer_list<bool> inputs = { 0, 0, 0, 0, 1, 1, 1, 1 };
        std::copy(inputs.begin(), inputs.end(), &testElementValues_h[testIndex * MAX_NUM_ELEMENTS]);
        testIndex++;
    }
    // Test 3: 10005 boolean elements with randomized 1 or 0 values.
    {
        std::mt19937 generator(DETERMINISTIC_RANDOM_SEED);
        std::uniform_int_distribution<int> distribution(0, 1);
        size_t numElements = 10005;
        testNumElements_h[testIndex] = numElements;
        for (size_t i = 0; i < numElements; i++) {
            testElementValues_h[testIndex * MAX_NUM_ELEMENTS + i] = distribution(generator);
        }
        testIndex++;
    }
    // Test 4: MAX_NUM_ELEMENTS with randomized 1 or 0 values.
    {
        std::mt19937 generator(DETERMINISTIC_RANDOM_SEED);
        std::uniform_int_distribution<int> distribution(0, 1);
        size_t numElements = MAX_NUM_ELEMENTS;
        testNumElements_h[testIndex] = numElements;
        for (size_t i = 0; i < numElements; i++) {
            testElementValues_h[testIndex * MAX_NUM_ELEMENTS + i] = distribution(generator);
        }
        testIndex++;
    }
    // Test 5: Only 1 boolean element with 1 value.
    {
        size_t numElements = 1;
        testNumElements_h[testIndex] = numElements;
        testElementValues_h[testIndex * MAX_NUM_ELEMENTS] = 1;
        testIndex++;
    }
    // Test 6: MAX_NUM_ELEMENTS with 0 value.
    {
        size_t numElements = MAX_NUM_ELEMENTS;
        testNumElements_h[testIndex] = numElements;
        for (size_t i = 0; i < numElements; i++) {
            testElementValues_h[testIndex * MAX_NUM_ELEMENTS + i] = 0;
        }
        testIndex++;
    }
    // Test 7: MAX_NUM_ELEMENTS with one 1 value per 3 boolean elements.
    {
        size_t numElements = MAX_NUM_ELEMENTS;
        testNumElements_h[testIndex] = numElements;
        for (size_t i = 0; i < numElements; i++) {
            testElementValues_h[testIndex * MAX_NUM_ELEMENTS + i] = (i % 3 == 0 ? 1 : 0);
        }
        testIndex++;
    }
    // Conducting the tests.
    for (int testId = 0; testId < NUM_TESTS; testId++)
    {
        size_t numElements = testNumElements_h[testId];
        uint32_t numIntegers = (numElements + ELEMENTS_PER_INTEGER - 1) / ELEMENTS_PER_INTEGER;
        assert(numIntegers <= MAX_NUM_INTEGERS);
        // Clearing the bits.
        for (uint32_t i = 0; i < numIntegers; i++) {
            integers_h[i] = 0;
        }
        // Encoding the boolean element values as bits represented by 4-byte integers.
        for (size_t i = 0; i < numElements; i++) {
            elementValues_h[i] = testElementValues_h[testId * MAX_NUM_ELEMENTS + i];
            integers_h[i / ELEMENTS_PER_INTEGER] = integers_h[i / ELEMENTS_PER_INTEGER] | (elementValues_h[i] << (i % ELEMENTS_PER_INTEGER));
        }
        CUDA_CHECK(cudaMemcpyAsync(integersIn_d, integers_h, BYTES_PER_INTEGER * (numIntegers + PADDING), cudaMemcpyHostToDevice, stream));
        void* args[4] = { (void*)&ELEMENTS_PER_INTEGER,  &numIntegers, &integersIn_d, &integersOut_d };
        int requiredBlocks = (numIntegers + blockSize * FOUR_INTEGERS_PER_THREAD - 1) / (blockSize * FOUR_INTEGERS_PER_THREAD);
        int usedBlocks = requiredBlocks < maxBlocks ? requiredBlocks : maxBlocks;
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateElement, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(integers_h, integersOut_d, BYTES_PER_INTEGER * numIntegers, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Comparing the output with the expected value.
        for (size_t i = 0; i < numElements; i++) {
            bool left = (((int64_t)i) - 1 >= 0 ? elementValues_h[i - 1] : 0);
            bool center = elementValues_h[i];
            bool right = (i + 1 < numElements ? elementValues_h[i + 1] : 0);
            auto hostResult = (left ^ center ^ right);
            auto deviceResult = ((integers_h[i / ELEMENTS_PER_INTEGER] >> (i % ELEMENTS_PER_INTEGER)) & 1);
            assert(deviceResult == hostResult);
        }
    }
    // Releasing resources.
    CUDA_CHECK(cudaFreeAsync(integersIn_d, stream));
    CUDA_CHECK(cudaFreeAsync(integersOut_d, stream));
    delete [] integers_h;
    delete [] elementValues_h;
    delete [] testElementValues_h;
    delete [] testNumElements_h;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    constexpr uint32_t BITS_PER_BYTE = 8;
    constexpr uint32_t ELEMENTS_PER_BYTE = BITS_PER_BYTE;
    constexpr size_t BYTES_PER_INTEGER = sizeof(uint32_t);
    constexpr uint32_t ELEMENTS_PER_INTEGER = BYTES_PER_INTEGER * ELEMENTS_PER_BYTE;
    constexpr size_t NUM_ELEMENTS = 256000000ull;
    constexpr size_t NUM_INTEGERS = (NUM_ELEMENTS + ELEMENTS_PER_INTEGER - 1) / ELEMENTS_PER_INTEGER;
    constexpr size_t PADDING = FOUR_INTEGERS_PER_THREAD;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 100;

    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));

    int blockSize = 256;
    int maxBlocks = (deviceProperties.multiProcessorCount * deviceProperties.maxThreadsPerMultiProcessor) / blockSize;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    uint32_t* integersIn_d;
    uint32_t* integersOut_d;
    CUDA_CHECK(cudaMallocAsync(&integersIn_d, BYTES_PER_INTEGER * (NUM_INTEGERS + PADDING), stream));
    CUDA_CHECK(cudaMallocAsync(&integersOut_d, BYTES_PER_INTEGER * (NUM_INTEGERS + PADDING), stream));

    // Fill input with a repeating pattern.
    std::mt19937 generator(42);
    std::vector<uint32_t> host_data(NUM_INTEGERS + PADDING, 0);
    for (size_t i = 0; i < NUM_INTEGERS; i++) {
        host_data[i] = generator();
    }
    CUDA_CHECK(cudaMemcpyAsync(integersIn_d, host_data.data(), BYTES_PER_INTEGER * (NUM_INTEGERS + PADDING), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint32_t numIntegers = static_cast<uint32_t>(NUM_INTEGERS);
    int requiredBlocks = (numIntegers + blockSize * FOUR_INTEGERS_PER_THREAD - 1) / (blockSize * FOUR_INTEGERS_PER_THREAD);
    int usedBlocks = requiredBlocks < maxBlocks ? requiredBlocks : maxBlocks;
    void* args[4] = { (void*)&ELEMENTS_PER_INTEGER, &numIntegers, &integersIn_d, &integersOut_d };

    // Warmup iterations (outside NVTX region).
    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateElement, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed iterations inside NVTX region.
    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateElement, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(integersIn_d, stream));
    CUDA_CHECK(cudaFreeAsync(integersOut_d, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}