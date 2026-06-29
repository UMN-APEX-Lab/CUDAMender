#undef NDEBUG
#include "radix_sort.h"
#include <nvtx3/nvToolsExt.h>
#include <cstring>

void benchmark() {
    constexpr int MAX_GRID_SIZE = 65535;
    constexpr int NUM_ELEMENTS = 131072;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 100;

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    assert(deviceProp.cooperativeLaunch);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const size_t bytes = static_cast<size_t>(NUM_ELEMENTS) * sizeof(unsigned int);

    unsigned int *keys_h = static_cast<unsigned int*>(malloc(bytes));
    assert(keys_h);
    srand(42);
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        keys_h[i] = rand() % 1000000;
    }

    unsigned int *keysIn_d, *keysOut_d, *globalHistograms_d;
    CUDA_CHECK(cudaMalloc(&keysIn_d, bytes));
    CUDA_CHECK(cudaMalloc(&keysOut_d, bytes));

    int numBlocks = (NUM_ELEMENTS + (BLOCK_SIZE * ITEMS_PER_THREAD) - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    int gridSize = std::min(numBlocks, MAX_GRID_SIZE);
    size_t histSize = (static_cast<size_t>(gridSize) * RADIX_SIZE + RADIX_SIZE) * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&globalHistograms_d, histSize));

    int numElementsCopy = NUM_ELEMENTS;
    void *args[] = {&keysIn_d, &keysOut_d, &globalHistograms_d, &numElementsCopy};

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaMemcpy(keysIn_d, keys_h, bytes, cudaMemcpyHostToDevice));
        cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(k_radixSort), dim3(gridSize), dim3(BLOCK_SIZE), args, 0, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Timed region
    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaMemcpy(keysIn_d, keys_h, bytes, cudaMemcpyHostToDevice));
        cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(k_radixSort), dim3(gridSize), dim3(BLOCK_SIZE), args, 0, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    CUDA_CHECK(cudaFree(keysIn_d));
    CUDA_CHECK(cudaFree(keysOut_d));
    CUDA_CHECK(cudaFree(globalHistograms_d));
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(keys_h);
}

void launch() {
    // Host-only constants
    constexpr int MAX_GRID_SIZE = 65535;
    constexpr unsigned int RANDOM_SEED = 12345;
    constexpr unsigned int MAX_RANDOM_VALUE = 1000000;

    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    assert(deviceCount > 0);

    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    assert(deviceProp.cooperativeLaunch);

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Test different sizes
    const int TEST_SIZES[] = {64, 256, 1024, 4096, 16384, 65536, 131072};
    const int NUM_TESTS = sizeof(TEST_SIZES) / sizeof(TEST_SIZES[0]);

    // Find maximum array size for memory allocation
    int maxNumElements = 0;
    for(int i = 0; i < NUM_TESTS; i++) {
        if(TEST_SIZES[i] > maxNumElements) {
            maxNumElements = TEST_SIZES[i];
        }
    }
    const size_t maxBytes = static_cast<size_t>(maxNumElements) * sizeof(unsigned int);

    // Allocate host memory once for maximum size
    unsigned int *keys_h = static_cast<unsigned int*>(malloc(maxBytes));
    unsigned int *cudaResult_h = static_cast<unsigned int*>(malloc(maxBytes));
    unsigned int *stdResult_h = static_cast<unsigned int*>(malloc(maxBytes));
    assert(keys_h && cudaResult_h && stdResult_h && "Host memory allocation failed!");

    // Allocate device memory once for maximum size
    unsigned int *keysIn_d;
    unsigned int *keysOut_d;
    CUDA_CHECK(cudaMallocAsync(&keysIn_d, maxBytes, stream));
    CUDA_CHECK(cudaMallocAsync(&keysOut_d, maxBytes, stream));

    // Seed and random initialization values
    srand(RANDOM_SEED);
    for(int test = 0; test < NUM_TESTS; test++) {
        const int numElements = TEST_SIZES[test];
        const size_t bytes = static_cast<size_t>(numElements) * sizeof(unsigned int);

        // Initialize test data
        for(int i = 0; i < numElements; i++) {
            keys_h[i] = rand() % MAX_RANDOM_VALUE;
        }

        // Copy original data for std::sort comparison
        memcpy(stdResult_h, keys_h, bytes);

        // Sort with std::sort for reference
        std::sort(stdResult_h, stdResult_h + numElements);

        // Copy data to device
        CUDA_CHECK(cudaMemcpyAsync(keysIn_d, keys_h, bytes, cudaMemcpyHostToDevice, stream));

        // Calculate grid dimensions for current array size
        int numBlocks = (numElements + (BLOCK_SIZE * ITEMS_PER_THREAD) - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
        int gridSize = std::min(numBlocks, MAX_GRID_SIZE);

        // Allocate global histogram storage once for the largest grid
        // (moved to host only once outside the loop if desired)
        static unsigned int *globalHistograms_d = nullptr;
        if (!globalHistograms_d) {
            size_t histSize = (static_cast<size_t>(MAX_GRID_SIZE) * RADIX_SIZE + RADIX_SIZE) * sizeof(unsigned int);
            CUDA_CHECK(cudaMallocAsync(&globalHistograms_d, histSize, stream));
        }

        // Launch cooperative kernel
        int numElementsCopy = numElements;
        void *args[] = {&keysIn_d, &keysOut_d, &globalHistograms_d, &numElementsCopy};
        cudaError_t err = cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(k_radixSort), dim3(gridSize), dim3(BLOCK_SIZE), args, 0, stream);
        assert(err == cudaSuccess);

        // Handle buffer swapping to ensure final result is in keysOut_d
        // After RADIX_PASSES (8, even number), result is in original src buffer (keysIn_d)
        // We need the final result in keysOut_d as the output buffer
        if(RADIX_PASSES % 2 == 0) {
            CUDA_CHECK(cudaMemcpyAsync(keysOut_d, keysIn_d, bytes, cudaMemcpyDeviceToDevice, stream));
        }

        // Copy result back from device (final result should be in keysOut_d)
        CUDA_CHECK(cudaMemcpyAsync(cudaResult_h, keysOut_d, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Validate results using assert
        for(int i = 0; i < numElements; i++) {
            assert(cudaResult_h[i] == stdResult_h[i]);
        }
    }

    // Cleanup all allocated memory
    CUDA_CHECK(cudaFreeAsync(keysIn_d, stream));
    CUDA_CHECK(cudaFreeAsync(keysOut_d, stream));
    cudaStreamSynchronize(stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(keys_h);
    free(cudaResult_h);
    free(stdResult_h);
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}