#undef NDEBUG
#include "kernel_contract.h"
#include <assert.h>
#include <stdio.h>
#include <cstring>
#include <algorithm>
#include <random>
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

// Maximum number of chunks allocated.
constexpr int MAX_CHUNKS = 1000;

// CUDA settings.
constexpr int NUM_THREADS_PER_BLOCK = 256;

// Test settings.
constexpr int NUM_TESTS = 7;
// Two samples have constant literal expected results ready, not needing computation on host.
constexpr int NUM_SAMPLES = 2;
constexpr float ERROR_TOLERANCE = 1e-5f;

void launch() {
    // Getting device properties.
    int deviceId = 0;
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));
    // Allocating stream.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Allocating host memory.
    int32_t * input_h = new int32_t[MAX_CHUNKS * CHUNK_ELEMENTS];
    float * output_h = new float[MAX_CHUNKS * CHUNK_ELEMENTS];
    // Allocating device memory.
    int32_t * input_d;
    float * output_d;
    CUDA_CHECK(cudaMallocAsync(&input_d, CHUNK_ELEMENTS * MAX_CHUNKS * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(input_d, 0, CHUNK_ELEMENTS * MAX_CHUNKS * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&output_d, CHUNK_ELEMENTS * MAX_CHUNKS * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(output_d, 0, CHUNK_ELEMENTS * MAX_CHUNKS * sizeof(float), stream));
    
    int * testNumChunks = new int[NUM_TESTS];
    int32_t * testInputs_h = new int32_t[NUM_TESTS * MAX_CHUNKS * CHUNK_ELEMENTS];
    float * testExpectedOutputs_h = new float[NUM_TESTS * MAX_CHUNKS * CHUNK_ELEMENTS];
    // Test 1: Sample 5 chunks with only 1 non-zero item per chunk, 1 chunk with all zero items.
    {
        int testIndex = 0;
        testNumChunks[testIndex] = 5;
        std::initializer_list<int32_t> inputs = { 0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0 };
        std::copy(inputs.begin(), inputs.end(), &testInputs_h[testIndex * MAX_CHUNKS * CHUNK_ELEMENTS]);
        std::initializer_list<float> expectedOutputs = { 
            -0.25f, -0.25f, -0.25f, 0.75f, 
            -0.5f, 1.5f, -0.5f, -0.5f, 
            3.0f - 0.5625f, -0.5625f, -0.5625f, -0.5625f, 
            -1.0f, -1.0f, 4.0f - 1.0f, -1.0f, 
            -1.0f, -1.0f, -1.0f, -1.0f 
        };
        std::copy(expectedOutputs.begin(), expectedOutputs.end(), &testExpectedOutputs_h[testIndex * MAX_CHUNKS * CHUNK_ELEMENTS]);
    }
    // Test 2: Sample 4 chunks with increasing elements.
    {
        int testIndex = 1;
        testNumChunks[testIndex] = 4;
        std::initializer_list<int32_t> inputs = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        std::copy(inputs.begin(), inputs.end(), &testInputs_h[testIndex * MAX_CHUNKS * CHUNK_ELEMENTS]);
        std::initializer_list<float> expectedOutputs = { 
            -2.2500f, -1.2500f, -0.2500f, 0.7500f, 
            -162.3750f, -161.3750f, -160.3750f, -159.3750f, 
            -849.3750f, -848.3750f, -847.3750f, -846.3750f, 
            -33203.0625f, -33202.0625f, -33201.0625f, -33200.0625f 
        };
        std::copy(expectedOutputs.begin(), expectedOutputs.end(), &testExpectedOutputs_h[testIndex * MAX_CHUNKS * CHUNK_ELEMENTS]);
    }
    // Test 3: randomized inputs in range (0, 10000)
    {
        int testIndex = 2;
        testNumChunks[testIndex] = MAX_CHUNKS;
        std::mt19937 generator(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int32_t> distribution(0, 10000);
        // Initializing the inputs.
        for(int i = 0; i < testNumChunks[testIndex] * CHUNK_ELEMENTS; i++) {
            testInputs_h[i + testIndex * MAX_CHUNKS * CHUNK_ELEMENTS] = distribution(generator);
        }
    }
    // Test 4: randomized inputs in range (10000, 100000)
    {
        int testIndex = 3;
        testNumChunks[testIndex] = MAX_CHUNKS;
        std::mt19937 generator(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int32_t> distribution(10000, 100000);
        // Initializing the inputs.
        for(int i = 0; i < testNumChunks[testIndex] * CHUNK_ELEMENTS; i++) {
            testInputs_h[i + testIndex * MAX_CHUNKS * CHUNK_ELEMENTS] = distribution(generator);
        }
    }
    // Test 5: randomized inputs in range (100000000, 1000000000)
    {
        int testIndex = 4;
        testNumChunks[testIndex] = MAX_CHUNKS;
        std::mt19937 generator(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int32_t> distribution(100000000, 1000000000);
        // Initializing the inputs.
        for(int i = 0; i < testNumChunks[testIndex] * CHUNK_ELEMENTS; i++) {
            testInputs_h[i + testIndex * MAX_CHUNKS * CHUNK_ELEMENTS] = distribution(generator);
        }
    }
    // Test 6: all elements are zero
    {
        int testIndex = 5;
        testNumChunks[testIndex] = MAX_CHUNKS;
        // Initializing the inputs.
        for(int i = 0; i < testNumChunks[testIndex] * CHUNK_ELEMENTS; i++) {
            testInputs_h[i + testIndex * MAX_CHUNKS * CHUNK_ELEMENTS] = 0;
        }
    }
    // Test 7: Negative elements.
    {
        int testIndex = 6;
        testNumChunks[testIndex] = MAX_CHUNKS;
        // Initializing the inputs.
        for(int i = 0; i < testNumChunks[testIndex] * CHUNK_ELEMENTS; i++) {
            testInputs_h[i + testIndex * MAX_CHUNKS * CHUNK_ELEMENTS] = -i;
        }
    }
    // Calculating the expected results for the non-sample tests.
    for(int test = NUM_SAMPLES; test < NUM_TESTS; test++) {
        int numChunks = testNumChunks[test];
        for(int i = 0; i < numChunks; i++) {
            int offset = test * MAX_CHUNKS * CHUNK_ELEMENTS;
            int32_t data1 = testInputs_h[offset + i * CHUNK_ELEMENTS];
            int32_t data2 = testInputs_h[offset + i * CHUNK_ELEMENTS + 1];
            int32_t data3 = testInputs_h[offset + i * CHUNK_ELEMENTS + 2];
            int32_t data4 = testInputs_h[offset + i * CHUNK_ELEMENTS + 3];
            int bitCounts[4] = { 0, 0, 0, 0};
            float average = data1 * 0.25f + data2 * 0.25f + data3 * 0.25f + data4 * 0.25f;
            for(int j = 0; j < 32; j++) {
                bitCounts[0] += ((data1 >> j) & 1u);
                bitCounts[1] += ((data2 >> j) & 1u);
                bitCounts[2] += ((data3 >> j) & 1u);
                bitCounts[3] += ((data4 >> j) & 1u);
            }
            int maxBits = (bitCounts[0] > bitCounts[1] ? bitCounts[0] : bitCounts[1]);
            maxBits = (maxBits > bitCounts[2] ? maxBits : bitCounts[2]);
            maxBits = (maxBits > bitCounts[3] ? maxBits : bitCounts[3]);
            testExpectedOutputs_h[offset + i * CHUNK_ELEMENTS] = data1 - powf(average, (float)maxBits);
            testExpectedOutputs_h[offset + i * CHUNK_ELEMENTS + 1] = data2 - powf(average, (float)maxBits);
            testExpectedOutputs_h[offset + i * CHUNK_ELEMENTS + 2] = data3 - powf(average, (float)maxBits);
            testExpectedOutputs_h[offset + i * CHUNK_ELEMENTS + 3] = data4 - powf(average, (float)maxBits);
        }
    }
    // Running the tests.
    for(int test = 0; test < NUM_TESTS; test++)
    {
        int offset = test * MAX_CHUNKS * CHUNK_ELEMENTS;
        int numChunks = testNumChunks[test];
        for(int i = 0; i < numChunks * CHUNK_ELEMENTS; i++) {
            input_h[i] = testInputs_h[i + offset];
        }
        // Copying the inputs to the device.
        size_t sizeOfInput = numChunks * CHUNK_ELEMENTS * sizeof(int32_t);
        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, sizeOfInput, cudaMemcpyHostToDevice, stream));
        // Calculating the result.
        void * args[3] = { &input_d, &output_d, &numChunks };
        int smCount = deviceProperties.multiProcessorCount;
        int smThreads = deviceProperties.maxThreadsPerMultiProcessor;
        int maxThreads = smCount * smThreads;
        int maxBlocks = maxThreads / NUM_THREADS_PER_BLOCK;
         int requiredThreads = (numChunks + CHUNK_ELEMENTS - 1) / CHUNK_ELEMENTS;
        int requiredBlocks = (requiredThreads + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
        int usedBlocks = requiredBlocks < maxBlocks ? requiredBlocks : maxBlocks;
        // Grid: (usedBlocks, 1, 1)
        // Block: (NUM_THREADS_PER_BLOCK, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateDifferencesFromPowerOfAveragePerChunk, 
                                    dim3(usedBlocks, 1, 1), 
                                    dim3(NUM_THREADS_PER_BLOCK, 1, 1), 
                                    args, 
                                    0, 
                                    stream));
        // Copying the outputs to the host.
        size_t sizeOfOutput = numChunks * CHUNK_ELEMENTS * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(output_h, output_d, sizeOfOutput, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Comparing results.
        for(int i = 0; i < numChunks; i++) {
            for(int j = 0; j < 4; j++) {
                float expected = testExpectedOutputs_h[offset + i * CHUNK_ELEMENTS + j];
                if(!isinf(expected) && !isinf(output_h[i * CHUNK_ELEMENTS + j])) {
                    if(fabs(expected) > 0.0f) {
                        assert(fabs((expected - output_h[i * CHUNK_ELEMENTS + j]) / expected) < ERROR_TOLERANCE);
                    } else {
                        assert(fabsf(expected - output_h[i * CHUNK_ELEMENTS + j]) < ERROR_TOLERANCE);
                    }
                }
            }
        }
    }
    
    // Deallocating device memory.
    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    // Deallocating host memory.
    delete [] input_h;
    delete [] output_h;
    delete [] testInputs_h;
    delete [] testNumChunks;
    delete [] testExpectedOutputs_h;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    int deviceId = 0;
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    constexpr int BENCH_CHUNKS = 2000000;
    constexpr int BENCH_ELEMENTS = BENCH_CHUNKS * CHUNK_ELEMENTS;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 100;

    int32_t *input_d;
    float *output_d;
    CUDA_CHECK(cudaMallocAsync(&input_d, BENCH_ELEMENTS * sizeof(int32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&output_d, BENCH_ELEMENTS * sizeof(float), stream));

    std::mt19937 generator(123);
    std::uniform_int_distribution<int32_t> distribution(0, 100000);
    int32_t *input_h = new int32_t[BENCH_ELEMENTS];
    for (int i = 0; i < BENCH_ELEMENTS; i++) {
        input_h[i] = distribution(generator);
    }
    CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, BENCH_ELEMENTS * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int smCount = deviceProperties.multiProcessorCount;
    int smThreads = deviceProperties.maxThreadsPerMultiProcessor;
    int maxThreads = smCount * smThreads;
    int maxBlocks = maxThreads / NUM_THREADS_PER_BLOCK;
    int numChunks = BENCH_CHUNKS;
    int requiredThreads = (numChunks + CHUNK_ELEMENTS - 1) / CHUNK_ELEMENTS;
    int requiredBlocks = (requiredThreads + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK;
    int usedBlocks = requiredBlocks < maxBlocks ? requiredBlocks : maxBlocks;
    void *args[3] = {&input_d, &output_d, &numChunks};

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void *)k_calculateDifferencesFromPowerOfAveragePerChunk,
                                    dim3(usedBlocks, 1, 1),
                                    dim3(NUM_THREADS_PER_BLOCK, 1, 1),
                                    args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void *)k_calculateDifferencesFromPowerOfAveragePerChunk,
                                    dim3(usedBlocks, 1, 1),
                                    dim3(NUM_THREADS_PER_BLOCK, 1, 1),
                                    args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete[] input_h;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}