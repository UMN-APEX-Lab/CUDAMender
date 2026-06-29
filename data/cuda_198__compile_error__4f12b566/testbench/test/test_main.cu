#undef NDEBUG
#include <assert.h>
#include <cstring>
#include <nvtx3/nvToolsExt.h>
#include "broadcast_tree.h"

void launch() {
    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));
    assert(deviceProperties.cooperativeLaunch);
    int warpSize = deviceProperties.warpSize;
    int maximumNumberOfOutputElements = MAXIMUM_NUMBER_OF_WARPS_TO_COMMUNICATE * warpSize;
    int maximumNumberOfMessageElements = MAXIMUM_NUMBER_OF_WARPS_TO_COMMUNICATE * warpSize;
    uint32_t * message_d;
    int * output_d;
    // Allocating the host buffer for output.
    int * output_h = new int[maximumNumberOfOutputElements];
    // Allocating and initializing device buffers for output and messaging.
    CUDA_CHECK(cudaMallocAsync(&message_d, sizeof(uint32_t) * maximumNumberOfMessageElements, stream));
    CUDA_CHECK(cudaMallocAsync(&output_d, sizeof(int) * maximumNumberOfOutputElements, stream));
    CUDA_CHECK(cudaMemsetAsync(message_d, 0, sizeof(uint32_t) * maximumNumberOfMessageElements, stream));
    CUDA_CHECK(cudaMemsetAsync(output_d, 0, sizeof(int) * maximumNumberOfOutputElements, stream));
    // Preparing the test data. The outputs should reflect these values + warp lane index.
    int testInputs[NUMBER_OF_TESTS] = { 1, 2, 10, 100, 1000, 25, 1000000 };
    int testNumCommunicatingThreads[NUMBER_OF_TESTS] = { 1, 2, 30, 2, 20, 500, 100 };

    // Iterating the tests.
    for(int test = 0; test < NUMBER_OF_TESTS; test++)
    {
        uint32_t input = testInputs[test];
        int numCommunicatingWarps = testNumCommunicatingThreads[test];
        int numBlocksPossiblePerSM;
        int numThreadsInBlock = 1024;
        int numSM = deviceProperties.multiProcessorCount;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(   &numBlocksPossiblePerSM,
                                                                    k_broadcastWithHierarchicalPath,
                                                                    numThreadsInBlock,
                                                                    0));
        int numBlocksRequired = (numCommunicatingWarps * warpSize + numThreadsInBlock - 1) / numThreadsInBlock;
        int numBlocksPossible = numSM * numBlocksPossiblePerSM;
        // The algorithm requires all warps to be in flight to function correctly.
        assert(numBlocksPossible >= numBlocksRequired);
        int numBlocksUsed = numBlocksRequired < numBlocksPossible ? numBlocksRequired : numBlocksPossible;

        // Grid: (numBlocksUsed, 1, 1)
        // Block: (numThreadsInBlock, 1, 1)
        dim3 gridDim(numBlocksUsed, 1, 1);
        dim3 blockDim(numThreadsInBlock, 1, 1);
        void * args[4] = { (void*)&input, (void*)&message_d, (void*)&numCommunicatingWarps, (void*)&output_d };
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_broadcastWithHierarchicalPath, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_h, output_d, sizeof(int) * maximumNumberOfOutputElements, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < numCommunicatingWarps * warpSize; i++) {
            int warpLane = (i % deviceProperties.warpSize);
            assert(output_h[i] == (input + warpLane));
        }
    }
    
    CUDA_CHECK(cudaFreeAsync(message_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    delete [] output_h;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));
    assert(deviceProperties.cooperativeLaunch);
    int warpSize = deviceProperties.warpSize;

    int numCommunicatingWarps = MAXIMUM_NUMBER_OF_WARPS_TO_COMMUNICATE;
    int numMessageElements = numCommunicatingWarps * warpSize;
    int numOutputElements = numCommunicatingWarps * warpSize;

    uint32_t * message_d;
    int * output_d;
    CUDA_CHECK(cudaMallocAsync(&message_d, sizeof(uint32_t) * numMessageElements, stream));
    CUDA_CHECK(cudaMallocAsync(&output_d, sizeof(int) * numOutputElements, stream));

    int numThreadsInBlock = 1024;
    int numSM = deviceProperties.multiProcessorCount;
    int numBlocksPossiblePerSM;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPossiblePerSM,
                                                              k_broadcastWithHierarchicalPath,
                                                              numThreadsInBlock, 0));
    int numBlocksRequired = (numCommunicatingWarps * warpSize + numThreadsInBlock - 1) / numThreadsInBlock;
    int numBlocksPossible = numSM * numBlocksPossiblePerSM;
    assert(numBlocksPossible >= numBlocksRequired);
    int numBlocksUsed = numBlocksRequired < numBlocksPossible ? numBlocksRequired : numBlocksPossible;

    dim3 gridDim(numBlocksUsed, 1, 1);
    dim3 blockDim(numThreadsInBlock, 1, 1);
    uint32_t input = 42;

    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 500;

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaMemsetAsync(message_d, 0, sizeof(uint32_t) * numMessageElements, stream));
        CUDA_CHECK(cudaMemsetAsync(output_d, 0, sizeof(int) * numOutputElements, stream));
        void * args[4] = { (void*)&input, (void*)&message_d, (void*)&numCommunicatingWarps, (void*)&output_d };
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_broadcastWithHierarchicalPath, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaMemsetAsync(message_d, 0, sizeof(uint32_t) * numMessageElements, stream));
        CUDA_CHECK(cudaMemsetAsync(output_d, 0, sizeof(int) * numOutputElements, stream));
        void * args[4] = { (void*)&input, (void*)&message_d, (void*)&numCommunicatingWarps, (void*)&output_d };
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_broadcastWithHierarchicalPath, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(message_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}