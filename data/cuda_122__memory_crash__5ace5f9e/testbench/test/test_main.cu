#include "merge_sort.h"
#include "cuda_common.h"
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>
#undef NDEBUG

void benchmark() {
    const int numElements = 512;
    const int WARMUP_ITERS = 3;
    const int TIMED_ITERS = 100;

    float *input_h = (float *)malloc(numElements * sizeof(float));
    srand(42);
    for (int i = 0; i < numElements; i++) {
        input_h[i] = (float)(rand() % 10000);
    }

    float *input_d, *output_d, *sortedBlocks_d;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMalloc(&input_d, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_d, numElements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sortedBlocks_d, numElements * sizeof(float)));

    int threadsPerBlock = BLOCKSIZE;
    dim3 blockSize(threadsPerBlock, 1, 1);
    dim3 gridSize((numElements + blockSize.x - 1) / blockSize.x, 1, 1);

    int numBlocksPerSm = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, k_mergeSort, threadsPerBlock, BLOCKSIZE * sizeof(float)));
    int maxGridSize = prop.multiProcessorCount * numBlocksPerSm;
    if ((int)gridSize.x > maxGridSize) {
        gridSize.x = maxGridSize;
    }

    void *args[] = {&input_d, &sortedBlocks_d, &output_d, const_cast<int*>(&numElements)};

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, numElements * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_mergeSort, gridSize, blockSize,
                                               args, BLOCKSIZE * sizeof(float), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h, numElements * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_mergeSort, gridSize, blockSize,
                                               args, BLOCKSIZE * sizeof(float), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    free(input_h);
    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    CUDA_CHECK(cudaFree(sortedBlocks_d));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void launch() {
    const int NUMTESTCASES = 7;
    int numElements[NUMTESTCASES] = {6, 8, 9, 10, 11, 15, 7};       // Number of input elements in the list
    int maxNumElements = *std::max_element(numElements, numElements + NUMTESTCASES);
    // Host input and output arrays
    float input_h[NUMTESTCASES][maxNumElements] ={  {6.0, 5.0, 4.0, 3.0, 2.0, 1.0},
                                                    {17.0, 1.0, 15.0, 3.0, 18.0, 2.0, 11.0, 12.0},
                                                    {5.0, 8.0, 2.0, 4.0, 10.0, 18.0, 9.0, 25.0, 1.0},
                                                    {25.0, 57.0, 2.0, 38.0, 49.0, 11.0, 79.0, 88.0, 5.0, 3.0},
                                                    {12.0, 32.0, 2.5, 1.3, 55.7, 38.2, 7.0, 15.5, 1.5, 22.5, 3.8},
                                                    {125, 133, 145, 5.8, 38.3, 55.7, 125, 133, 77.5, 33.4, 55.7, 88.6, 77.5, 4.2, 2.0},
                                                    {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0}};
    float expectedOutput_h[NUMTESTCASES][maxNumElements] = {{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                                            {1.0, 2.0, 3.0, 11.0, 12.0, 15.0, 17.0, 18.0},
                                                            {1.0, 2.0, 4.0, 5.0, 8.0, 9.0, 10.0, 18.0, 25.0},
                                                            {2.0, 3.0, 5.0, 11.0, 25.0, 38.0, 49.0, 57.0, 79.0, 88.0},
                                                            {1.3, 1.5, 2.5, 3.8, 7.0, 12.0, 15.5, 22.5, 32.0, 38.2, 55.7},
                                                            {2.0, 4.2, 5.8, 33.4, 38.3, 55.7, 55.7, 77.5, 77.5, 88.6, 125, 125, 133, 133, 145},
                                                            {1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0}};
    float *output_h = (float *) calloc(maxNumElements, sizeof(float)); 
    
    // Device input and output pointers
    float *input_d;
    float *output_d;
    float *sortedBlocks_d;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Allocate device memory
    CUDA_CHECK(cudaMallocAsync((void**)&input_d, maxNumElements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&output_d, maxNumElements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&sortedBlocks_d, maxNumElements * sizeof(float), stream));

    // Define block and grid sizes
    int threadsPerBlock = BLOCKSIZE;

    // Blocks: (BLOCKSIZE, 1, 1)
    dim3 blockSize(threadsPerBlock, 1, 1);

    for(int tc = 0; tc < NUMTESTCASES; tc++){
        // Calculate grid size per test case based on actual numElements
        dim3 gridSize((numElements[tc] + blockSize.x - 1) / blockSize.x, 1, 1);
        // Copy input data to device
        CUDA_CHECK(cudaMemcpyAsync(input_d, 
                                   input_h[tc], 
                                   numElements[tc] * sizeof(float), 
                                   cudaMemcpyHostToDevice, 
                                   stream));

        // Launch the mergeSort kernel to sort the elements with in the block
        void *args[] = {&input_d, &sortedBlocks_d, &output_d, &numElements[tc]};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_mergeSort, gridSize, blockSize, args, BLOCKSIZE * sizeof(float), stream));
        
        // Copy the output back to host
        CUDA_CHECK(cudaMemcpyAsync(output_h, 
                                   output_d, 
                                   numElements[tc] * sizeof(float), 
                                   cudaMemcpyDeviceToHost, 
                                   stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        //validate the results
        for(int i = 0; i < numElements[tc]; i++) {
            assert(fabs(output_h[i] - expectedOutput_h[tc][i]) < TOLERANCE);
        }
    }

    // Free host and device memory
    free(output_h);    
    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    CUDA_CHECK(cudaFreeAsync(sortedBlocks_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}