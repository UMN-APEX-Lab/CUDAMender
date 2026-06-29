#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <random>
#include <cfloat>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nvtx3/nvToolsExt.h>
#include <string>
#include "k_sortSegments.h"
#include "cuda_helpers.h"

void launch() {
    // Algorithm-related constants.
    constexpr int MAXIMUM_NUMBER_OF_SUB_ARRAYS = 10000;
    constexpr int MAXIMUM_SIZE_OF_SUB_ARRAY = 128;
    constexpr int MAXIMUM_ARRAY_SIZE = MAXIMUM_SIZE_OF_SUB_ARRAY * MAXIMUM_NUMBER_OF_SUB_ARRAYS;
    constexpr int DETERMINISTIC_RANDOM_SEED = 42;
    int deviceId = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, deviceId));
    int numThreadsPerBlock = 1000;
    int numWarpsPerBlock = (numThreadsPerBlock + properties.warpSize - 1) / properties.warpSize;
    int maximumNumBlocks = properties.maxBlocksPerMultiProcessor * properties.multiProcessorCount;
    // Input array.
    float *array_h = new float[MAXIMUM_ARRAY_SIZE];
    // Output array.
    float *arrayOut_h = new float[MAXIMUM_ARRAY_SIZE];
    float *array_d;
    float *arrayOut_d;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocAsync(&array_d, MAXIMUM_ARRAY_SIZE * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&arrayOut_d, MAXIMUM_ARRAY_SIZE * sizeof(float), stream));
    auto hToD = cudaMemcpyHostToDevice;
    auto dToH = cudaMemcpyDeviceToHost;
    
    // Test 1: segment size = 5, number of segments = 3
    {
        int segmentSize = 5;
        int numSegments = 3;
        int arraySize = numSegments * segmentSize;
        int numWarpsRequired = numSegments;
        int numBlocksRequired = (numWarpsRequired + numWarpsPerBlock - 1) / numWarpsPerBlock;
        int numBlocksUsed = (maximumNumBlocks < numBlocksRequired ? maximumNumBlocks : numBlocksRequired);
        auto initializerList = { 10.2f, -11.0f, -25.3f, 35.0f, -448.0f, -5.0f, -68.0f, -57.0f, -128.0f, -99955.0f, -20.0f, -211.0f, -312.0f, -0.1f, -14.5f };
        std::copy(initializerList.begin(), initializerList.end(), array_h);
        // Copying array to device.
        CUDA_CHECK(cudaMemcpyAsync(array_d, array_h, sizeof(float) * arraySize, hToD, stream));
        void * args[4] = { (void*)&array_d, (void*)&arrayOut_d, (void*)&segmentSize, (void*)&arraySize };
        dim3 gridDim(numBlocksUsed, 1, 1);
        dim3 blockDim(numThreadsPerBlock, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)&k_sortSegments, gridDim, blockDim, args, 0, stream));
        // Copying result to host.
        CUDA_CHECK(cudaMemcpyAsync(arrayOut_h, arrayOut_d, sizeof(float) * arraySize, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int index = 0;
        for(int i = 0; i < numSegments; i++) {
            for(int j = 0; j < segmentSize - 1; j++) {
                assert(arrayOut_h[index] < std::nextafter(arrayOut_h[index + 1], fabs(arrayOut_h[index + 1]) * 2.0f));
                index++;
            }
            index++;
        }
    }
    
    // Test 2: segment size = 10, number of segments = 4
    {
        int segmentSize = 10;
        int numSegments = 4;
        int arraySize = numSegments * segmentSize;
        int numWarpsRequired = numSegments;
        int numBlocksRequired = (numWarpsRequired + numWarpsPerBlock - 1) / numWarpsPerBlock;
        int numBlocksUsed = (maximumNumBlocks < numBlocksRequired ? maximumNumBlocks : numBlocksRequired);
        auto initializerList = { 1230.0f, -123.0f, -200.0f, -300.0f, -4.1f, -5.2f, -6.7f, 7.8f, 8.1f, -19.0f, -190.0f, -1199.0f, -412.0f, -153.0f, -174.0f, -715.0f, -176.0f, -177.0f, -718.0f, -179.0f, -207.0f, -821.0f, -282.0f, -238.0f, -924.0f, -325.0f, -236.0f, -273.0f, -248.0f, -249.0f, -304.0f, -531.0f, -325.0f, -353.0f, -3400.0f, -3005.0f, -306.0f, -370.0f, -398.0f, -399.0f };
        std::copy(initializerList.begin(), initializerList.end(), array_h);
        // Copying array to device.
        CUDA_CHECK(cudaMemcpyAsync(array_d, array_h, sizeof(float) * arraySize, hToD, stream));
        void * args[4] = { (void*)&array_d, (void*)&arrayOut_d, (void*)&segmentSize, (void*)&arraySize };
        dim3 gridDim(numBlocksUsed, 1, 1);
        dim3 blockDim(numThreadsPerBlock, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)&k_sortSegments, gridDim, blockDim, args, 0, stream));
        // Copying result to host.
        CUDA_CHECK(cudaMemcpyAsync(arrayOut_h, arrayOut_d, sizeof(float) * arraySize, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int index = 0;
        for(int i = 0; i < numSegments; i++) {
            for(int j = 0; j < segmentSize - 1; j++) {
                assert(arrayOut_h[index] < std::nextafter(arrayOut_h[index + 1], fabs(arrayOut_h[index + 1]) * 2.0f));
                index++;
            }
            index++;
        }
    }
    
    // Test 3: segment size equals the maximum allowed size, number of segments equals the maximum allowed number, normalized random values.
    {
        int segmentSize = 11;
        int numSegments = 1;
        int arraySize = numSegments * segmentSize;
        int numWarpsRequired = numSegments;
        int numBlocksRequired = (numWarpsRequired + numWarpsPerBlock - 1) / numWarpsPerBlock;
        int numBlocksUsed = (maximumNumBlocks < numBlocksRequired ? maximumNumBlocks : numBlocksRequired);
        std::mt19937 generator(DETERMINISTIC_RANDOM_SEED);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        for(int i = 0; i < arraySize; i++) {
            array_h[i] = distribution(generator);
        }
        // Copying array to device.
        CUDA_CHECK(cudaMemcpyAsync(array_d, array_h, sizeof(float) * arraySize, hToD, stream));
        void * args[4] = { (void*)&array_d, (void*)&arrayOut_d, (void*)&segmentSize, (void*)&arraySize };
        dim3 gridDim(numBlocksUsed, 1, 1);
        dim3 blockDim(numThreadsPerBlock, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)&k_sortSegments, gridDim, blockDim, args, 0, stream));
        // Copying result to host.
        CUDA_CHECK(cudaMemcpyAsync(arrayOut_h, arrayOut_d, sizeof(float) * arraySize, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int index = 0;
        for(int i = 0; i < numSegments; i++) {
            for(int j = 0; j < segmentSize - 1; j++) {
                assert(arrayOut_h[index] < std::nextafter(arrayOut_h[index + 1], fabs(arrayOut_h[index + 1]) * 2.0f));
                index++;
            }
            index++;
        }
    }
    
    // Test 4: segment size = 2, number of segments = 1000, large random values.
    {
        int segmentSize = 2;
        int numSegments = 1000;
        int arraySize = numSegments * segmentSize;
        int numWarpsRequired = numSegments;
        int numBlocksRequired = (numWarpsRequired + numWarpsPerBlock - 1) / numWarpsPerBlock;
        int numBlocksUsed = (maximumNumBlocks < numBlocksRequired ? maximumNumBlocks : numBlocksRequired);
        std::mt19937 generator(DETERMINISTIC_RANDOM_SEED);
        std::uniform_real_distribution<float> distribution(-1e30, 1e30);
        for(int i = 0; i < arraySize; i++) {
            array_h[i] = distribution(generator);
        }
        // Copying array to device.
        CUDA_CHECK(cudaMemcpyAsync(array_d, array_h, sizeof(float) * arraySize, hToD, stream));
        void * args[4] = { (void*)&array_d, (void*)&arrayOut_d, (void*)&segmentSize, (void*)&arraySize };
        dim3 gridDim(numBlocksUsed, 1, 1);
        dim3 blockDim(numThreadsPerBlock, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)&k_sortSegments, gridDim, blockDim, args, 0, stream));
        // Copying result to host.
        CUDA_CHECK(cudaMemcpyAsync(arrayOut_h, arrayOut_d, sizeof(float) * arraySize, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int index = 0;
        for(int i = 0; i < numSegments; i++) {
            for(int j = 0; j < segmentSize - 1; j++) {
                assert(arrayOut_h[index] < std::nextafter(arrayOut_h[index + 1], fabs(arrayOut_h[index + 1]) * 2.0f));
                index++;
            }
            index++;
        }
    }
    
    // Test 5: segment size = 100, number of segments = 1500, inputs are randomly shuffled.
    {
        int segmentSize = 100;
        int numSegments = 5000;
        int arraySize = numSegments * segmentSize;
        int numWarpsRequired = numSegments;
        int numBlocksRequired = (numWarpsRequired + numWarpsPerBlock - 1) / numWarpsPerBlock;
        int numBlocksUsed = (maximumNumBlocks < numBlocksRequired ? maximumNumBlocks : numBlocksRequired);
        int index = 0;
        for(int i = 0; i < numSegments; i++) {
            for(int j = 0; j < segmentSize; j++) {
                array_h[index++] = -j;
            }
            std::shuffle(array_h + index - segmentSize, array_h + index, std::default_random_engine(i));
        }
        // Copying array to device.
        CUDA_CHECK(cudaMemcpyAsync(array_d, array_h, sizeof(float) * arraySize, hToD, stream));
        void * args[4] = { (void*)&array_d, (void*)&arrayOut_d, (void*)&segmentSize, (void*)&arraySize };
        dim3 gridDim(numBlocksUsed, 1, 1);
        dim3 blockDim(numThreadsPerBlock, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)&k_sortSegments, gridDim, blockDim, args, 0, stream));
        // Copying result to host.
        CUDA_CHECK(cudaMemcpyAsync(arrayOut_h, arrayOut_d, sizeof(float) * arraySize, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        index = 0;
        for(int i = 0; i < numSegments; i++) {
            for(int j = 0; j < segmentSize - 1; j++) {
                assert(arrayOut_h[index] < std::nextafter(arrayOut_h[index + 1], fabs(arrayOut_h[index + 1]) * 2.0f));
                index++;
            }
            index++;
        }
    }
    
    // Test 6: segment size = 1, number of segments = 10, unique values equal to index + PI.
    {
        int segmentSize = 1;
        int numSegments = 10;
        int arraySize = numSegments * segmentSize;
        int numWarpsRequired = numSegments;
        int numBlocksRequired = (numWarpsRequired + numWarpsPerBlock - 1) / numWarpsPerBlock;
        int numBlocksUsed = (maximumNumBlocks < numBlocksRequired ? maximumNumBlocks : numBlocksRequired);
        constexpr float PI = 3.1415f;
        for(int i = 0; i < numSegments; i++) {
            array_h[i] = PI + cos(i);
        }
        // Copying array to device.
        CUDA_CHECK(cudaMemcpyAsync(array_d, array_h, sizeof(float) * arraySize, hToD, stream));
        void * args[4] = { (void*)&array_d, (void*)&arrayOut_d, (void*)&segmentSize, (void*)&arraySize };
        dim3 gridDim(numBlocksUsed, 1, 1);
        dim3 blockDim(numThreadsPerBlock, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)&k_sortSegments, gridDim, blockDim, args, 0, stream));
        // Copying result to host.
        CUDA_CHECK(cudaMemcpyAsync(arrayOut_h, arrayOut_d, sizeof(float) * arraySize, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int index = 0;
        for(int i = 0; i < numSegments; i++) {
            for(int j = 0; j < segmentSize - 1; j++) {
                assert(arrayOut_h[index] < std::nextafter(arrayOut_h[index + 1], fabs(arrayOut_h[index + 1]) * 2.0f));
                index++;
            }
            index++;
        }
    }
    
    // Test 7: segment size equals the maximum allowed size, the number of segments equals the maximum allowed number, with one infinity value for each segment.
    {
        int segmentSize = MAXIMUM_SIZE_OF_SUB_ARRAY;
        int numSegments = MAXIMUM_NUMBER_OF_SUB_ARRAYS;
        int arraySize = numSegments * segmentSize;
        int numWarpsRequired = numSegments;
        int numBlocksRequired = (numWarpsRequired + numWarpsPerBlock - 1) / numWarpsPerBlock;
        int numBlocksUsed = (maximumNumBlocks < numBlocksRequired ? maximumNumBlocks : numBlocksRequired);
        for(int i = 0; i < arraySize; i++) {
            array_h[i] = ((i % segmentSize == 0) ? std::numeric_limits<float>::infinity() : sin(i));
        }
        // Copying array to device.
        CUDA_CHECK(cudaMemcpyAsync(array_d, array_h, sizeof(float) * arraySize, hToD, stream));
        void * args[4] = { (void*)&array_d, (void*)&arrayOut_d, (void*)&segmentSize, (void*)&arraySize };
        dim3 gridDim(numBlocksUsed, 1, 1);
        dim3 blockDim(numThreadsPerBlock, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)&k_sortSegments, gridDim, blockDim, args, 0, stream));
        // Copying result to host.
        CUDA_CHECK(cudaMemcpyAsync(arrayOut_h, arrayOut_d, sizeof(float) * arraySize, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int index = 0;
        for(int i = 0; i < numSegments; i++) {
            for(int j = 0; j < segmentSize - 1; j++) {
                assert(arrayOut_h[index] < std::nextafter(arrayOut_h[index + 1], fabs(arrayOut_h[index + 1]) * 2.0f));
                index++;
            }
            index++;
        }
    }
    
    CUDA_CHECK(cudaFreeAsync(array_d, stream));
    CUDA_CHECK(cudaFreeAsync(arrayOut_d, stream));
    delete [] array_h;
    delete [] arrayOut_h;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void run_benchmark() {
    constexpr int SEGMENT_SIZE = 128;
    constexpr int NUM_SEGMENTS = 10000;
    constexpr int ARRAY_SIZE = SEGMENT_SIZE * NUM_SEGMENTS;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 100;

    int deviceId = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, deviceId));
    int numThreadsPerBlock = 1000;
    int numWarpsPerBlock = (numThreadsPerBlock + properties.warpSize - 1) / properties.warpSize;
    int maximumNumBlocks = properties.maxBlocksPerMultiProcessor * properties.multiProcessorCount;
    int numWarpsRequired = NUM_SEGMENTS;
    int numBlocksRequired = (numWarpsRequired + numWarpsPerBlock - 1) / numWarpsPerBlock;
    int numBlocksUsed = (maximumNumBlocks < numBlocksRequired ? maximumNumBlocks : numBlocksRequired);

    float *array_h = new float[ARRAY_SIZE];
    std::mt19937 generator(123);
    std::uniform_real_distribution<float> distribution(-1e6f, 1e6f);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        array_h[i] = distribution(generator);
    }

    float *array_d, *arrayOut_d;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocAsync(&array_d, ARRAY_SIZE * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&arrayOut_d, ARRAY_SIZE * sizeof(float), stream));
    CUDA_CHECK(cudaMemcpyAsync(array_d, array_h, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int segmentSize = SEGMENT_SIZE;
    int arraySize = ARRAY_SIZE;
    void *args[4] = { (void*)&array_d, (void*)&arrayOut_d, (void*)&segmentSize, (void*)&arraySize };
    dim3 grid(numBlocksUsed, 1, 1);
    dim3 block(numThreadsPerBlock, 1, 1);

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)&k_sortSegments, grid, block, args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)&k_sortSegments, grid, block, args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(array_d, stream));
    CUDA_CHECK(cudaFreeAsync(arrayOut_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete[] array_h;
}

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "--perf") {
        run_benchmark();
    } else {
        launch();
    }
}