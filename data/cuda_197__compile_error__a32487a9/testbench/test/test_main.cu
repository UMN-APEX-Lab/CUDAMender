#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <cstring>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include "kernel_contract.h"

void launch() {
    constexpr int NUM_TESTS = 7;
    constexpr float ERROR_TOLERANCE = 1e-4f;
    int deviceIndex = 0;
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceIndex));
    int blockSize = 256;
    int maxBlocks = (deviceProperties.maxThreadsPerMultiProcessor * deviceProperties.multiProcessorCount) / blockSize;

    int numWarpsPerBlock = blockSize / deviceProperties.warpSize;
    int maxSegmentsPerPass = maxBlocks * numWarpsPerBlock;
    int NUM_MAX_SEGMENTS = maxSegmentsPerPass * 2;
    int NUM_MAX_ELEMENTS = SEGMENT_SIZE * NUM_MAX_SEGMENTS;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Allocating host buffers.
    float* array_h = new float[NUM_MAX_ELEMENTS];
    float* testArray_h = new float[NUM_TESTS * NUM_MAX_ELEMENTS];
    int* testNumSegments_h = new int[NUM_TESTS];
    float* testThresholds_h = new float[NUM_TESTS];
    float* testDefaultValues = new float[NUM_TESTS];
    
    // Allocating device buffers.
    float* array_d;
    CUDA_CHECK(cudaMallocAsync(&array_d, sizeof(float) * NUM_MAX_ELEMENTS, stream));

    int testIndex = 0;
    // Test 1
    {
        int numSegments = 1;
        testNumSegments_h[testIndex] = numSegments;
        testThresholds_h[testIndex] = 0.5f;
        testDefaultValues[testIndex] = 0.0f;
        std::initializer_list<float> data = {
            0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        std::copy(data.begin(), data.end(), &testArray_h[testIndex * NUM_MAX_ELEMENTS]);
        testIndex++;
    }
    // Test 2
    {
        int numSegments = 1;
        testNumSegments_h[testIndex] = numSegments;
        testThresholds_h[testIndex] = 0.0f;
        testDefaultValues[testIndex] = 0.0f;
        std::initializer_list<float> data = {
            -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
        };
        std::copy(data.begin(), data.end(), &testArray_h[testIndex * NUM_MAX_ELEMENTS]);
        testIndex++;
    }
    // Test 3: All segments, mixed values
    {
        int numSegments = NUM_MAX_SEGMENTS;
        testNumSegments_h[testIndex] = numSegments;
        testThresholds_h[testIndex] = 0.0f;
        testDefaultValues[testIndex] = 0.0f;
        for (int i = 0; i < numSegments * SEGMENT_SIZE; i++) {
            testArray_h[testIndex * NUM_MAX_ELEMENTS + i] = i - numSegments * SEGMENT_SIZE * 0.5f;
        }
        testIndex++;
    }
    // Test 4: All segments, all above threshold
    {
        int numSegments = NUM_MAX_SEGMENTS;
        testNumSegments_h[testIndex] = numSegments;
        testThresholds_h[testIndex] = -1.0f;
        testDefaultValues[testIndex] = 0.0f;
        for (int i = 0; i < numSegments * SEGMENT_SIZE; i++) {
            testArray_h[testIndex * NUM_MAX_ELEMENTS + i] = 1.0f;
        }
        testIndex++;
    }
    // Test 5: All segments, all below threshold
    {
        int numSegments = NUM_MAX_SEGMENTS;
        testNumSegments_h[testIndex] = numSegments;
        testThresholds_h[testIndex] = 1.0f;
        testDefaultValues[testIndex] = 0.0f;
        for (int i = 0; i < numSegments * SEGMENT_SIZE; i++) {
            testArray_h[testIndex * NUM_MAX_ELEMENTS + i] = -1.0f;
        }
        testIndex++;
    }
    // Test 6: All segments, NaN handling
    {
        int numSegments = NUM_MAX_SEGMENTS;
        testNumSegments_h[testIndex] = numSegments;
        testThresholds_h[testIndex] = 3.14f;
        testDefaultValues[testIndex] = 0.0f;
        for (int i = 0; i < numSegments * SEGMENT_SIZE; i++) {
            testArray_h[testIndex * NUM_MAX_ELEMENTS + i] = ((i % 2) ? NAN : 3.1415f);
        }
        testIndex++;
    }
    // Test 7: All segments, modular pattern
    {
        int numSegments = NUM_MAX_SEGMENTS;
        testNumSegments_h[testIndex] = numSegments;
        testThresholds_h[testIndex] = 50.0f;
        testDefaultValues[testIndex] = 0.0f;
        for (int i = 0; i < numSegments * SEGMENT_SIZE; i++) {
            testArray_h[testIndex * NUM_MAX_ELEMENTS + i] = (i % 100);
        }
        testIndex++;
    }
    for (int test = 0; test < testIndex; test++)
    {
        int numSegments = testNumSegments_h[test];
        float threshold = testThresholds_h[test];
        float defaultValue = testDefaultValues[test];
        int numElements = numSegments * SEGMENT_SIZE;
        for (int i = 0; i < numElements; i++) {
            array_h[i] = testArray_h[test * NUM_MAX_ELEMENTS + i];
        }
        CUDA_CHECK(cudaMemcpyAsync(array_d, array_h, sizeof(float) * numElements, cudaMemcpyHostToDevice, stream));
        void* args[4] = { &numSegments, &array_d, &threshold, &defaultValue };
        int requiredBlocks = (numSegments + numWarpsPerBlock - 1) / numWarpsPerBlock;
        int usedBlocks = requiredBlocks < maxBlocks ? requiredBlocks : maxBlocks;
        int sharedMem = blockSize * sizeof(float);
        CUDA_CHECK(cudaLaunchKernel((void*)k_compactElementsOfSegmentsWithThreshold, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, sharedMem, stream));
        CUDA_CHECK(cudaMemcpyAsync(array_h, array_d, sizeof(float) * numElements, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < numSegments; i++) {
            int compactIndex = 0;
            // Identifying compacted elements.
            for (int j = 0; j < SEGMENT_SIZE; j++) {
                float testData = testArray_h[test * NUM_MAX_ELEMENTS + i * SEGMENT_SIZE + j];
                if (testData > threshold) {
                    assert(fabsf(array_h[compactIndex + i * SEGMENT_SIZE] - testData) < ERROR_TOLERANCE);
                    compactIndex++;
                }
            }
            // Identifying default elements.
            for (; compactIndex < SEGMENT_SIZE; compactIndex++) {
                assert(fabsf(array_h[compactIndex + i * SEGMENT_SIZE] - defaultValue) < ERROR_TOLERANCE);
            }
        }
    }
    
    // Deallocating device buffers.
    CUDA_CHECK(cudaFreeAsync(array_d, stream));
    // Deallocating host buffers.
    delete [] array_h;
    delete [] testArray_h;
    delete [] testNumSegments_h;
    delete [] testThresholds_h;
    delete [] testDefaultValues;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    constexpr int NUM_SEGMENTS = 1000000;
    constexpr int NUM_ELEMENTS = SEGMENT_SIZE * NUM_SEGMENTS;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 100;
    constexpr float THRESHOLD = 0.5f;
    constexpr float DEFAULT_VALUE = 0.0f;

    int deviceIndex = 0;
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceIndex));
    int blockSize = 256;
    int maxBlocks = (deviceProperties.maxThreadsPerMultiProcessor * deviceProperties.multiProcessorCount) / blockSize;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float* host_src = new float[NUM_ELEMENTS];
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        host_src[i] = (i % 100) * 0.05f;
    }

    float* array_d;
    CUDA_CHECK(cudaMallocAsync(&array_d, sizeof(float) * NUM_ELEMENTS, stream));

    int numWarpsPerBlock = blockSize / deviceProperties.warpSize;
    int requiredBlocks = (NUM_SEGMENTS + numWarpsPerBlock - 1) / numWarpsPerBlock;
    int usedBlocks = requiredBlocks < maxBlocks ? requiredBlocks : maxBlocks;
    int sharedMem = blockSize * sizeof(float);

    int numSegments = NUM_SEGMENTS;
    float threshold = THRESHOLD;
    float defaultValue = DEFAULT_VALUE;
    void* args[4] = { &numSegments, &array_d, &threshold, &defaultValue };

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaMemcpyAsync(array_d, host_src, sizeof(float) * NUM_ELEMENTS, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_compactElementsOfSegmentsWithThreshold, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, sharedMem, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaMemcpyAsync(array_d, host_src, sizeof(float) * NUM_ELEMENTS, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_compactElementsOfSegmentsWithThreshold, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, sharedMem, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(array_d, stream));
    delete[] host_src;
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