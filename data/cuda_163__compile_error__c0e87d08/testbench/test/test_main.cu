#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <nvtx3/nvToolsExt.h>
#include "lcg_kernel.h"

#define CUDA_CHECK(call) {                                      \
    cudaError_t error = call;                                  \
    if(error != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

void launch() {
    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));   

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Allocating host buffers.
    uint32_t * data_h = new uint32_t[MAXIMUM_ARRAY_LENGTH];
    uint32_t * expectedData_h = new uint32_t[MAXIMUM_ARRAY_LENGTH];
    // Allocating device buffer.
    uint32_t * data_d;
    CUDA_CHECK(cudaMallocAsync(&data_d, MAXIMUM_BUFFER_BYTES, stream));
    // Initializing all elements including padding to zero.
    CUDA_CHECK(cudaMemsetAsync(data_d, 0, MAXIMUM_BUFFER_BYTES, stream));
    auto hToD = cudaMemcpyHostToDevice;
    auto dToH = cudaMemcpyDeviceToHost;



    // Test 1: 10 elements with sequentially increasing values.
    {
        int numElements = 10;
        assert(numElements <= MAXIMUM_ARRAY_LENGTH);
        for(int i = 0; i < numElements; i++) {
            data_h[i] = i;
            uint64_t data = data_h[i];
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            expectedData_h[i] = data;
        }
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, numElements * sizeof(uint32_t), hToD, stream));

        void * args[2] = { &data_d, &numElements };
        // Dynamically calculating the shared memory size used to not surpassed the limitations of the hardware.
        int sizeOfSharedMemoryRequiredPerThread = sizeof(uint4) * NUMBER_OF_SHARED_MEMORY_ARRAYS_PER_BLOCK;
        int maximumSharedMemoryPerBlockAvailable = prop.sharedMemPerBlock;
        int maximumThreadsPerBlockAvailable = maximumSharedMemoryPerBlockAvailable / sizeOfSharedMemoryRequiredPerThread;
        int threadsPerBlockUsed = std::min(maximumThreadsPerBlockAvailable, MAXIMUM_NUMBER_OF_THREADS_PER_BLOCK_ALLOWED);
        int sizeOfSharedMemoryRequired = sizeOfSharedMemoryRequiredPerThread * threadsPerBlockUsed;
        // Dynamically calculating the number of blocks of the grid.
        int maxBlocks = (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / threadsPerBlockUsed;
        int elementsComputedPerBlockPerIteration = threadsPerBlockUsed * MEMORY_ACCESS_WIDTH;
        int blocksRequired = 1 + (numElements - 1) / elementsComputedPerBlockPerIteration;
        int blocksUsed = (blocksRequired <= maxBlocks ? blocksRequired : maxBlocks);

        // Grid: (1, 1, 1)
        // Block: (1024, 1, 1)
        dim3 gridDim(blocksUsed, 1, 1);
        dim3 blockDim(threadsPerBlockUsed, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLCG, gridDim, blockDim, args, sizeOfSharedMemoryRequired, stream));
        CUDA_CHECK(cudaMemcpyAsync(data_h, data_d, numElements * sizeof(uint32_t), dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        for(int i = 0; i < numElements; i++) {
            assert(data_h[i] == expectedData_h[i]);
        }
    }
    // Test 2: 9 elements with sequentially decreasing values.
    {
        int numElements = 9;
        assert(numElements <= MAXIMUM_ARRAY_LENGTH);
        for(int i = 0; i < numElements; i++) {
            data_h[i] = 150 - i;
            uint64_t data = data_h[i];
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            expectedData_h[i] = data;
        }
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, numElements * sizeof(uint32_t), hToD, stream));

        void * args[2] = { &data_d, &numElements };
        // Dynamically calculating the shared memory size used to not surpassed the limitations of the hardware.
        int sizeOfSharedMemoryRequiredPerThread = sizeof(uint4) * NUMBER_OF_SHARED_MEMORY_ARRAYS_PER_BLOCK;
        int maximumSharedMemoryPerBlockAvailable = prop.sharedMemPerBlock;
        int maximumThreadsPerBlockAvailable = maximumSharedMemoryPerBlockAvailable / sizeOfSharedMemoryRequiredPerThread;
        int threadsPerBlockUsed = std::min(maximumThreadsPerBlockAvailable, MAXIMUM_NUMBER_OF_THREADS_PER_BLOCK_ALLOWED);
        int sizeOfSharedMemoryRequired = sizeOfSharedMemoryRequiredPerThread * threadsPerBlockUsed;
        // Dynamically calculating the number of blocks of the grid.
        int maxBlocks = (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / threadsPerBlockUsed;
        int elementsComputedPerBlockPerIteration = threadsPerBlockUsed * MEMORY_ACCESS_WIDTH;
        int blocksRequired = 1 + (numElements - 1) / elementsComputedPerBlockPerIteration;
        int blocksUsed = (blocksRequired <= maxBlocks ? blocksRequired : maxBlocks);

        // Grid: (1, 1, 1)
        // Block: (1024, 1, 1)
        dim3 gridDim(blocksUsed, 1, 1);
        dim3 blockDim(threadsPerBlockUsed, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLCG, gridDim, blockDim, args, sizeOfSharedMemoryRequired, stream));
        CUDA_CHECK(cudaMemcpyAsync(data_h, data_d, numElements * sizeof(uint32_t), dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        for(int i = 0; i < numElements; i++) {
            assert(data_h[i] == expectedData_h[i]);
        }
    }
    // Test 3: 10000 elements with only 5th bit set.
    {
        int numElements = 10000;
        assert(numElements <= MAXIMUM_ARRAY_LENGTH);
        for(int i = 0; i < numElements; i++) {
            data_h[i] = (1 << 4);
            uint64_t data = data_h[i];
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            expectedData_h[i] = data;
        }
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, numElements * sizeof(uint32_t), hToD, stream));
        
        void * args[2] = { &data_d, &numElements };
        // Dynamically calculating the shared memory size used to not surpassed the limitations of the hardware.
        int sizeOfSharedMemoryRequiredPerThread = sizeof(uint4) * NUMBER_OF_SHARED_MEMORY_ARRAYS_PER_BLOCK;
        int maximumSharedMemoryPerBlockAvailable = prop.sharedMemPerBlock;
        int maximumThreadsPerBlockAvailable = maximumSharedMemoryPerBlockAvailable / sizeOfSharedMemoryRequiredPerThread;
        int threadsPerBlockUsed = std::min(maximumThreadsPerBlockAvailable, MAXIMUM_NUMBER_OF_THREADS_PER_BLOCK_ALLOWED);
        int sizeOfSharedMemoryRequired = sizeOfSharedMemoryRequiredPerThread * threadsPerBlockUsed;
        // Dynamically calculating the number of blocks of the grid.
        int maxBlocks = (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / threadsPerBlockUsed;
        int elementsComputedPerBlockPerIteration = threadsPerBlockUsed * MEMORY_ACCESS_WIDTH;
        int blocksRequired = 1 + (numElements - 1) / elementsComputedPerBlockPerIteration;
        int blocksUsed = (blocksRequired <= maxBlocks ? blocksRequired : maxBlocks);

        // Grid: (3, 1, 1)
        // Block: (1024, 1, 1)
        dim3 gridDim(blocksUsed, 1, 1);
        dim3 blockDim(threadsPerBlockUsed, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLCG, gridDim, blockDim, args, sizeOfSharedMemoryRequired, stream));
        CUDA_CHECK(cudaMemcpyAsync(data_h, data_d, numElements * sizeof(uint32_t), dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        for(int i = 0; i < numElements; i++) {
            assert(data_h[i] == expectedData_h[i]);
        }
    }
    // Test 4: 100 elements with sequentially changing bits set.
    {
        int numElements = 100;
        assert(numElements <= MAXIMUM_ARRAY_LENGTH);
        for(int i = 0; i < numElements; i++) {
            data_h[i] = (1 << (i % 32));
            uint64_t data = data_h[i];
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            expectedData_h[i] = data;
        }
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, numElements * sizeof(uint32_t), hToD, stream));
        
        void * args[2] = { &data_d, &numElements };
        // Dynamically calculating the shared memory size used to not surpassed the limitations of the hardware.
        int sizeOfSharedMemoryRequiredPerThread = sizeof(uint4) * NUMBER_OF_SHARED_MEMORY_ARRAYS_PER_BLOCK;
        int maximumSharedMemoryPerBlockAvailable = prop.sharedMemPerBlock;
        int maximumThreadsPerBlockAvailable = maximumSharedMemoryPerBlockAvailable / sizeOfSharedMemoryRequiredPerThread;
        int threadsPerBlockUsed = std::min(maximumThreadsPerBlockAvailable, MAXIMUM_NUMBER_OF_THREADS_PER_BLOCK_ALLOWED);
        int sizeOfSharedMemoryRequired = sizeOfSharedMemoryRequiredPerThread * threadsPerBlockUsed;
        // Dynamically calculating the number of blocks of the grid.
        int maxBlocks = (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / threadsPerBlockUsed;
        int elementsComputedPerBlockPerIteration = threadsPerBlockUsed * MEMORY_ACCESS_WIDTH;
        int blocksRequired = 1 + (numElements - 1) / elementsComputedPerBlockPerIteration;
        int blocksUsed = (blocksRequired <= maxBlocks ? blocksRequired : maxBlocks);

        // Grid: (1, 1, 1)
        // Block: (1024, 1, 1)
        dim3 gridDim(blocksUsed, 1, 1);
        dim3 blockDim(threadsPerBlockUsed, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLCG, gridDim, blockDim, args, sizeOfSharedMemoryRequired, stream));
        CUDA_CHECK(cudaMemcpyAsync(data_h, data_d, numElements * sizeof(uint32_t), dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        for(int i = 0; i < numElements; i++) {
            assert(data_h[i] == expectedData_h[i]);
        }
    }
    // Test 5: MAXIMUM_ARRAY_LENGTH elements with index % 1000 value.
    {
        int numElements = MAXIMUM_ARRAY_LENGTH;
        assert(numElements <= MAXIMUM_ARRAY_LENGTH);
        for(int i = 0; i < numElements; i++) {
            data_h[i] = i % 1000;
            uint64_t data = data_h[i];
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            expectedData_h[i] = data;
        }
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, numElements * sizeof(uint32_t), hToD, stream));
        
        void * args[2] = { &data_d, &numElements };
        // Dynamically calculating the shared memory size used to not surpassed the limitations of the hardware.
        int sizeOfSharedMemoryRequiredPerThread = sizeof(uint4) * NUMBER_OF_SHARED_MEMORY_ARRAYS_PER_BLOCK;
        int maximumSharedMemoryPerBlockAvailable = prop.sharedMemPerBlock;
        int maximumThreadsPerBlockAvailable = maximumSharedMemoryPerBlockAvailable / sizeOfSharedMemoryRequiredPerThread;
        int threadsPerBlockUsed = std::min(maximumThreadsPerBlockAvailable, MAXIMUM_NUMBER_OF_THREADS_PER_BLOCK_ALLOWED);
        int sizeOfSharedMemoryRequired = sizeOfSharedMemoryRequiredPerThread * threadsPerBlockUsed;
        // Dynamically calculating the number of blocks of the grid.
        int maxBlocks = (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / threadsPerBlockUsed;
        int elementsComputedPerBlockPerIteration = threadsPerBlockUsed * MEMORY_ACCESS_WIDTH;
        int blocksRequired = 1 + (numElements - 1) / elementsComputedPerBlockPerIteration;
        int blocksUsed = (blocksRequired <= maxBlocks ? blocksRequired : maxBlocks);

        // Grid: (25, 1, 1)
        // Block: (1024, 1, 1)
        dim3 gridDim(blocksUsed, 1, 1);
        dim3 blockDim(threadsPerBlockUsed, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLCG, gridDim, blockDim, args, sizeOfSharedMemoryRequired, stream));
        CUDA_CHECK(cudaMemcpyAsync(data_h, data_d, numElements * sizeof(uint32_t), dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        for(int i = 0; i < numElements; i++) {
            assert(data_h[i] == expectedData_h[i]);
        }
    }
    // Test 6: MAXIMUM_ARRAY_LENGTH elements with 0xFFFFFFFF value.
    {
        int numElements = MAXIMUM_ARRAY_LENGTH;
        assert(numElements <= MAXIMUM_ARRAY_LENGTH);
        for(int i = 0; i < numElements; i++) {
            data_h[i] = 0xFFFFFFFF;
            uint64_t data = data_h[i];
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            expectedData_h[i] = data;
        }
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, numElements * sizeof(uint32_t), hToD, stream));
        
        void * args[2] = { &data_d, &numElements };
        // Dynamically calculating the shared memory size used to not surpassed the limitations of the hardware.
        int sizeOfSharedMemoryRequiredPerThread = sizeof(uint4) * NUMBER_OF_SHARED_MEMORY_ARRAYS_PER_BLOCK;
        int maximumSharedMemoryPerBlockAvailable = prop.sharedMemPerBlock;
        int maximumThreadsPerBlockAvailable = maximumSharedMemoryPerBlockAvailable / sizeOfSharedMemoryRequiredPerThread;
        int threadsPerBlockUsed = std::min(maximumThreadsPerBlockAvailable, MAXIMUM_NUMBER_OF_THREADS_PER_BLOCK_ALLOWED);
        int sizeOfSharedMemoryRequired = sizeOfSharedMemoryRequiredPerThread * threadsPerBlockUsed;
        // Dynamically calculating the number of blocks of the grid.
        int maxBlocks = (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / threadsPerBlockUsed;
        int elementsComputedPerBlockPerIteration = threadsPerBlockUsed * MEMORY_ACCESS_WIDTH;
        int blocksRequired = 1 + (numElements - 1) / elementsComputedPerBlockPerIteration;
        int blocksUsed = (blocksRequired <= maxBlocks ? blocksRequired : maxBlocks);

        // Grid: (25, 1, 1)
        // Block: (1024, 1, 1)
        dim3 gridDim(blocksUsed, 1, 1);
        dim3 blockDim(threadsPerBlockUsed, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLCG, gridDim, blockDim, args, sizeOfSharedMemoryRequired, stream));
        CUDA_CHECK(cudaMemcpyAsync(data_h, data_d, numElements * sizeof(uint32_t), dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        for(int i = 0; i < numElements; i++) {
            assert(data_h[i] == expectedData_h[i]);
        }
    }
    // Test 7: 55 elements with 0 value.
    {
        int numElements = 55;
        assert(numElements <= MAXIMUM_ARRAY_LENGTH);
        for(int i = 0; i < numElements; i++) {
            data_h[i] = 0;
            uint64_t data = data_h[i];
            for(int step = 0; step < LCG_STEPS; step++) {
                data = (data * LCG_MULTIPLIER + LCG_OFFSET) % LCG_MODULUS;
            }
            expectedData_h[i] = data;
        }
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, numElements * sizeof(uint32_t), hToD, stream));
        
        void * args[2] = { &data_d, &numElements };
        // Dynamically calculating the shared memory size used to not surpassed the limitations of the hardware.
        int sizeOfSharedMemoryRequiredPerThread = sizeof(uint4) * NUMBER_OF_SHARED_MEMORY_ARRAYS_PER_BLOCK;
        int maximumSharedMemoryPerBlockAvailable = prop.sharedMemPerBlock;
        int maximumThreadsPerBlockAvailable = maximumSharedMemoryPerBlockAvailable / sizeOfSharedMemoryRequiredPerThread;
        int threadsPerBlockUsed = std::min(maximumThreadsPerBlockAvailable, MAXIMUM_NUMBER_OF_THREADS_PER_BLOCK_ALLOWED);
        int sizeOfSharedMemoryRequired = sizeOfSharedMemoryRequiredPerThread * threadsPerBlockUsed;
        // Dynamically calculating the number of blocks of the grid.
        int maxBlocks = (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / threadsPerBlockUsed;
        int elementsComputedPerBlockPerIteration = threadsPerBlockUsed * MEMORY_ACCESS_WIDTH;
        int blocksRequired = 1 + (numElements - 1) / elementsComputedPerBlockPerIteration;
        int blocksUsed = (blocksRequired <= maxBlocks ? blocksRequired : maxBlocks);

        // Grid: (1, 1, 1)
        // Block: (1024, 1, 1)
        dim3 gridDim(blocksUsed, 1, 1);
        dim3 blockDim(threadsPerBlockUsed, 1, 1);
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLCG, gridDim, blockDim, args, sizeOfSharedMemoryRequired, stream));
        CUDA_CHECK(cudaMemcpyAsync(data_h, data_d, numElements * sizeof(uint32_t), dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        for(int i = 0; i < numElements; i++) {
            assert(data_h[i] == expectedData_h[i]);
        }
    }
    CUDA_CHECK(cudaFreeAsync(data_d, stream));
    delete [] data_h;
    delete [] expectedData_h;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int numElements = MAXIMUM_ARRAY_LENGTH;
    uint32_t * data_h = new uint32_t[MAXIMUM_ARRAY_LENGTH];
    uint32_t * data_d;
    CUDA_CHECK(cudaMallocAsync(&data_d, MAXIMUM_BUFFER_BYTES, stream));

    int sizeOfSharedMemoryRequiredPerThread = sizeof(uint4) * NUMBER_OF_SHARED_MEMORY_ARRAYS_PER_BLOCK;
    int maximumSharedMemoryPerBlockAvailable = prop.sharedMemPerBlock;
    int maximumThreadsPerBlockAvailable = maximumSharedMemoryPerBlockAvailable / sizeOfSharedMemoryRequiredPerThread;
    int threadsPerBlockUsed = std::min(maximumThreadsPerBlockAvailable, MAXIMUM_NUMBER_OF_THREADS_PER_BLOCK_ALLOWED);
    int sizeOfSharedMemoryRequired = sizeOfSharedMemoryRequiredPerThread * threadsPerBlockUsed;
    int maxBlocks = (prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount) / threadsPerBlockUsed;
    int elementsComputedPerBlockPerIteration = threadsPerBlockUsed * MEMORY_ACCESS_WIDTH;
    int blocksRequired = 1 + (numElements - 1) / elementsComputedPerBlockPerIteration;
    int blocksUsed = (blocksRequired <= maxBlocks ? blocksRequired : maxBlocks);

    dim3 gridDim(blocksUsed, 1, 1);
    dim3 blockDim(threadsPerBlockUsed, 1, 1);

    for(int i = 0; i < numElements; i++) {
        data_h[i] = i % 1000;
    }

    auto hToD = cudaMemcpyHostToDevice;
    void * args[2] = { &data_d, &numElements };

    for(int warmup = 0; warmup < 3; warmup++) {
        CUDA_CHECK(cudaMemsetAsync(data_d, 0, MAXIMUM_BUFFER_BYTES, stream));
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, numElements * sizeof(uint32_t), hToD, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLCG, gridDim, blockDim, args, sizeOfSharedMemoryRequired, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for(int iter = 0; iter < 100; iter++) {
        CUDA_CHECK(cudaMemsetAsync(data_d, 0, MAXIMUM_BUFFER_BYTES, stream));
        CUDA_CHECK(cudaMemcpyAsync(data_d, data_h, numElements * sizeof(uint32_t), hToD, stream));
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateLCG, gridDim, blockDim, args, sizeOfSharedMemoryRequired, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(data_d, stream));
    delete [] data_h;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if(argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}