#include "bitonic_sort.h"
#include "common.h"
#include <algorithm>
#include <assert.h>
#include <cstdlib>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

#undef NDEBUG

void launch() {
    // Number of test cases
    const int TEST_CASE_COUNT = 8;
    // Sizes of the image in each test case
    const int INPUT_DATA_LENGTH[TEST_CASE_COUNT] = {5, 6, 5, 6, 7, 5, 6, 6};
    // Find the maximum image size
    const int MAX_VECTOR_SIZE = *std::max_element(INPUT_DATA_LENGTH, INPUT_DATA_LENGTH + TEST_CASE_COUNT);

    // Input vectors and configurations for the tests
    float inputImage[TEST_CASE_COUNT][MAX_VECTOR_SIZE] =  {
        {4.5, 3, 7.2, 5, 2.1},
        {12, 9.5, 15.3, 3, 5.8, 8},
        {6.2, 10, 1.5, 7, 8.1},
        {5, 4.2, 9.8, 3, 7.6, 2},
        {9, 1.1, 3, 5.5, 6.8, 10, 7.3},
        {13.5, 15, 7.2, 2.3, 9.9},
        {14, 3.4, 1, 8.7, 5.3, 7.1},
        {2, 7.4, 4.6, 6, 5, 3.1}
    };
    
    // expected outputs
    float expectedOutputData[TEST_CASE_COUNT][MAX_VECTOR_SIZE] =  {
        {2.1, 3, 4.5, 5, 7.2}, 
        {3, 5.8, 8, 9.5, 12, 15.3}, 
        {1.5, 6.2, 7, 8.1, 10}, 
        {2, 3, 4.2, 5, 7.6, 9.8}, 
        {1.1, 3, 5.5, 6.8, 7.3, 9, 10}, 
        {2.3, 7.2, 9.9, 13.5, 15}, 
        {1, 3.4, 5.3, 7.1, 8.7, 14}, 
        {2, 3.1, 4.6, 5, 6, 7.4}
    };

    // Use a CUDA stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize results on the host
    float *imageData_h;
    imageData_h = (float*)malloc(MAX_VECTOR_SIZE * sizeof(float));

    // Pointers for device memory (GPU)
    float *imageData_d;

    // Allocate the memory on the device
    CUDA_CHECK(cudaMallocAsync(&imageData_d, MAX_VECTOR_SIZE * sizeof(float), stream));

    // Loop to execute each test case
    for (int i = 0; i < TEST_CASE_COUNT; ++i) {
        int dataLength = INPUT_DATA_LENGTH[i];
        // Copy input data to the device
        CUDA_CHECK(cudaMemcpyAsync(imageData_d, inputImage[i], dataLength * sizeof(float), cudaMemcpyHostToDevice, stream));
        
        // Determine the number of threads and blocks
        dim3 gridSize = dim3((dataLength + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        dim3 blockSize = dim3(BLOCK_SIZE, 1, 1);

        // Execute the kernel
        // Grid:  (1, 1, 1)
        // Block: (256, 1, 1)
        void *args[] = {&imageData_d, (void*)&dataLength};
        CUDA_CHECK(cudaLaunchKernel((void*)k_bitonicSort, gridSize, blockSize, args, sizeof(float) * BLOCK_SIZE, stream));

        // Copy the result back to the host (CPU)
        CUDA_CHECK(cudaMemcpyAsync(imageData_h, imageData_d, dataLength * sizeof(float), cudaMemcpyDeviceToHost, stream));

        // Check tasks in the stream has completed
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Verify if the sorted data matches the expected output
        for (int j = 0; j < dataLength; j++) {
            assert(imageData_h[j] == expectedOutputData[i][j]);
        }
    }
    // Free device memory and stream
    CUDA_CHECK(cudaFreeAsync(imageData_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    // Free host memories
    free(imageData_h);
}

void benchmark() {
    // Large workload: 4096 blocks * 256 elements = 1,048,576 elements
    // Each block independently sorts its 256-element chunk
    const int NUM_BLOCKS = 4096;
    const int TOTAL_SIZE = NUM_BLOCKS * BLOCK_SIZE;
    const int WARMUP_ITERS = 3;
    const int TIMED_ITERS = 100;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float *h_data = (float*)malloc(TOTAL_SIZE * sizeof(float));
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, TOTAL_SIZE * sizeof(float)));

    // Fill with descending data so every block has work to do
    for (int i = 0; i < TOTAL_SIZE; i++) {
        h_data[i] = (float)(TOTAL_SIZE - i);
    }

    dim3 gridSize(NUM_BLOCKS, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);

    // Warmup iterations (outside NVTX region)
    for (int iter = 0; iter < WARMUP_ITERS; iter++) {
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
        void *args[] = {&d_data, (void*)&TOTAL_SIZE};
        CUDA_CHECK(cudaLaunchKernel((void*)k_bitonicSort, gridSize, blockSize, args, sizeof(float) * BLOCK_SIZE, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Timed iterations inside NVTX region
    nvtxRangePushA("bench_region");
    for (int iter = 0; iter < TIMED_ITERS; iter++) {
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, TOTAL_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
        void *args[] = {&d_data, (void*)&TOTAL_SIZE};
        CUDA_CHECK(cudaLaunchKernel((void*)k_bitonicSort, gridSize, blockSize, args, sizeof(float) * BLOCK_SIZE, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(h_data);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}