#include "miller_rabin.h"
#include <nvtx3/nvToolsExt.h>
#include <cstdlib>
#include <cstring>

static void benchmarkMillerRabin() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    gWarpSize = prop.warpSize;

    const int N = 131072;
    const int WARMUP = 3;
    const int TIMED = 100;

    std::vector<unsigned long long> h_input(N);
    for (int i = 0; i < N; i++) {
        h_input[i] = (unsigned long long)(i * 6 + 5) | 1ULL;
        if (h_input[i] < 5) h_input[i] = 5;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    unsigned long long *d_input, *d_output;
    size_t bytes = N * sizeof(unsigned long long);
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    int optBlockSize, blocksPerGrid;
    float theoreticalOccupancy;
    CUDA_CHECK(getOptimalLaunchParamsMillerRabin(N, optBlockSize, blocksPerGrid, theoreticalOccupancy));
    int threadsPerBlock = (optBlockSize < gWarpSize) ? gWarpSize : optBlockSize;
    size_t sharedMemSize = dynamicSMemSizeMillerRabin(threadsPerBlock);

    for (int i = 0; i < WARMUP; i++) {
        CUDA_CHECK(cudaMemsetAsync(d_output, 0, bytes, stream));
        void* args[] = { &d_input, &d_output, (void*)&N };
        CUDA_CHECK(cudaLaunchKernel((void*)k_millerRabin,
            dim3(blocksPerGrid, 1, 1), dim3(threadsPerBlock, 1, 1),
            args, sharedMemSize, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED; i++) {
        CUDA_CHECK(cudaMemsetAsync(d_output, 0, bytes, stream));
        void* args[] = { &d_input, &d_output, (void*)&N };
        CUDA_CHECK(cudaLaunchKernel((void*)k_millerRabin,
            dim3(blocksPerGrid, 1, 1), dim3(threadsPerBlock, 1, 1),
            args, sharedMemSize, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void launch() {

    // Retrieve device properties to get the warp size.
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    gWarpSize = prop.warpSize;
    
    // Input data
    unsigned long long inputNumbers_h[NUMBER_OF_TESTS][NUMBER_OF_ELEMENTS] = {
        {17, 561, 19, 7919, 23, 1009, 10007, 15},
        {29, 35, 31, 10037, 13, 999331, 7919, 77},
        {97, 561, 73, 103, 113, 199, 2003, 25},
        {101, 91, 89, 1009, 1021, 17, 71, 143},
        {211, 221, 223, 227, 229, 233, 239, 561},
        {307, 401, 509, 601, 701, 803, 907, 999},
        {997, 991, 983, 977, 971, 967, 953, 947},
        {7877, 8011, 8089, 8093, 8081, 7873, 7817, 561},
        {997, 991, 983, 977, 971, 967, 953, 947, 7877, 8011, 8089, 8093, 8081, 7873, 7817, 561}
    };

    unsigned long long expectedResults[NUMBER_OF_TESTS][NUMBER_OF_ELEMENTS] = {
        {17, 0, 19, 7919, 23, 1009, 10007, 0},   
        {29, 0, 31, 10037, 13, 999331, 7919, 0}, 
        {97, 0, 73, 103, 113, 199, 2003, 0},    
        {101, 0, 89, 1009, 1021, 17, 71, 0},    
        {211, 0, 223, 227, 229, 233, 239, 0},    
        {307, 401, 509, 601, 701, 0, 907, 0},      
        {997, 991, 983, 977, 971, 967, 953, 947},
        {7877, 8011, 8089, 8093, 8081, 7873, 7817, 0},
        {997, 991, 983, 977, 971, 967, 953, 947,7877, 8011, 8089, 8093, 8081, 7873, 7817, 0}
    };

    // Create single CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocate device memory once for the maximum list size.
    size_t testListSize = NUMBER_OF_ELEMENTS * sizeof(unsigned long long);
    unsigned long long *inputNumbers_d, *primalityResults_d;
    CUDA_CHECK(cudaMallocAsync(&inputNumbers_d, testListSize * NUMBER_OF_TESTS, stream));
    CUDA_CHECK(cudaMallocAsync(&primalityResults_d, testListSize * NUMBER_OF_TESTS, stream));

    // Lambda to launch the kernel with the optimal configuration.
    auto launchKernelWithConfig = [&stream](unsigned long long* currentInput, unsigned long long* currentOutput,
                                              int numTests, int threadsPerBlock, int blocksPerGrid) {
        void* args[] = { &currentInput, &currentOutput, &numTests };
        size_t sharedMemSize = dynamicSMemSizeMillerRabin(threadsPerBlock);
        CUDA_CHECK(cudaLaunchKernel((void*)k_millerRabin, dim3(blocksPerGrid, 1, 1), dim3(threadsPerBlock, 1, 1), args, sharedMemSize, stream));
    };
    
    // Lambda to run an individual test.
    auto runTest = [&](int listIndex, int numTests) {
        unsigned long long* currentInput = inputNumbers_d + (listIndex * NUMBER_OF_ELEMENTS);
        unsigned long long* currentOutput = primalityResults_d + (listIndex * NUMBER_OF_ELEMENTS);
        
        // Copy the current test input to device.
        CUDA_CHECK(cudaMemcpyAsync(currentInput, inputNumbers_h[listIndex],
                                   testListSize, cudaMemcpyHostToDevice, stream));
        // Clear the output area.
        CUDA_CHECK(cudaMemsetAsync(currentOutput, 0, testListSize, stream));
        
        // Determine optimal launch parameters.
        int optBlockSize, blocksPerGrid;
        float theoreticalOccupancy;
        CUDA_CHECK(getOptimalLaunchParamsMillerRabin(numTests, optBlockSize, blocksPerGrid, theoreticalOccupancy));
        
        // Ensure threadsPerBlock is at least the warp size.
        int threadsPerBlock = (optBlockSize < gWarpSize) ? gWarpSize : optBlockSize;
        
        // Launch the kernel.
        launchKernelWithConfig(currentInput, currentOutput, numTests, threadsPerBlock, blocksPerGrid);
        
        // Copy the results back to host.
        std::vector<unsigned long long> primalityResults_h(numTests, 0);
        CUDA_CHECK(cudaMemcpyAsync(primalityResults_h.data(), currentOutput, testListSize, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Verify the results.
        for (int i = 0; i < numTests; i++) {
            assert(primalityResults_h[i] == expectedResults[listIndex][i]);
        }
    };
    
    // Process each test list.
    for (int listIndex = 0; listIndex < NUMBER_OF_TESTS; listIndex++) {
        runTest(listIndex, NUMBER_OF_ELEMENTS);
    }
    
    // Cleanup device resources.
    CUDA_CHECK(cudaFreeAsync(inputNumbers_d, stream));
    CUDA_CHECK(cudaFreeAsync(primalityResults_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmarkMillerRabin();
    } else {
        launch();
    }
    return 0;
}