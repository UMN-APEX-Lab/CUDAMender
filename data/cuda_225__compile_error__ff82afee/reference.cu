#include "least_squares.h"
#include "least_squares_common.h"

void run(int numElements, int maxInputSize, 
         cudaStream_t* stream, cudaEvent_t firstStreamStop, cudaEvent_t secondStreamStop, 
         int* observedValues_d, int* predictedValues_d, 
         int* interimBufferObservedVal_d, int* interimBufferPredictedVal_d, float* averageObservedValue_d, 
         float* averagePredictedValue_d, float* summationOfProducts_d, float* summationOfSquares_d, 
         float* interimBufferSumOfProducts_d, float* interimBufferSumOfSquares_d, float* outputSlope_d, 
         float* outputIntercept_d) {
    const int NUMBER_OF_THREADS_PER_BLOCK = 256;
    const int LAST_KERNEL_NUMBER_OF_THREADS_PER_BLOCK = 1;
    const int LAST_KERNEL_NUMBER_OF_BLOCKS = 1; 
    
    // Synchronize streams on device to have visibility of latest data on both streams before starting computations.
    CUDA_CHECK(cudaEventRecord(firstStreamStop, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaStreamWaitEvent(stream[SECOND_STREAM_INDEX], firstStreamStop));
    
    // Reset output while distributing the work to both streams.
    CUDA_CHECK(cudaMemsetAsync(averageObservedValue_d, SET_TO_ZERO, OUTPUT_SIZE * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMemsetAsync(averagePredictedValue_d, SET_TO_ZERO, OUTPUT_SIZE * sizeof(float), stream[SECOND_STREAM_INDEX]));
    CUDA_CHECK(cudaMemsetAsync(summationOfProducts_d, SET_TO_ZERO, OUTPUT_SIZE * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMemsetAsync(summationOfSquares_d, SET_TO_ZERO, OUTPUT_SIZE * sizeof(float), stream[SECOND_STREAM_INDEX]));
    CUDA_CHECK(cudaMemsetAsync(outputSlope_d, SET_TO_ZERO, OUTPUT_SIZE * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMemsetAsync(outputIntercept_d, SET_TO_ZERO, OUTPUT_SIZE * sizeof(float), stream[SECOND_STREAM_INDEX]));
    CUDA_CHECK(cudaMemsetAsync(interimBufferObservedVal_d, SET_TO_ZERO, maxInputSize * sizeof(int), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMemsetAsync(interimBufferPredictedVal_d, SET_TO_ZERO, maxInputSize * sizeof(int), stream[SECOND_STREAM_INDEX]));
    CUDA_CHECK(cudaMemsetAsync(interimBufferSumOfProducts_d, SET_TO_ZERO, maxInputSize * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMemsetAsync(interimBufferSumOfSquares_d, SET_TO_ZERO, maxInputSize * sizeof(float), stream[SECOND_STREAM_INDEX]));
    
    // Get device properties.
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    // Set kernel configuration.
    int numThreadsPerBlock = NUMBER_OF_THREADS_PER_BLOCK;
    numThreadsPerBlock = min(numThreadsPerBlock, prop.maxThreadsDim[0]);
    
    int numBlocks = ceil((float)(numElements) / numThreadsPerBlock);
    // Ensure numBlocks is even to support grid reduction.
    if(numBlocks % 2 != 0) {
        numBlocks += 1;
    }
    numBlocks = min(numBlocks, prop.maxGridSize[0]);
    
    dim3 block(numThreadsPerBlock, 1, 1);
    dim3 grid(numBlocks, 1, 1);
    
    //Launch Kernels in different Streams
    //Grid: (numElements / 256, 1, 1)
    //Block: (256, 1, 1)
    //First Step: launch the k_computeAverageObservedValue kernel in first stream
    void *argsAvgObservedValueKernel[] = {&observedValues_d, &predictedValues_d, &averageObservedValue_d, &interimBufferObservedVal_d, &numElements};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_computeAverageObservedValue, grid, block, argsAvgObservedValueKernel, numThreadsPerBlock * sizeof(int), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaEventRecord(firstStreamStop, stream[FIRST_STREAM_INDEX]));
    
    //First Step: launch the k_computeAveragePredictedValue kernel in second stream
    void *argsAveragePredictedValueKernel[] = {&observedValues_d, &predictedValues_d, &averagePredictedValue_d, &interimBufferPredictedVal_d, &numElements};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_computeAveragePredictedValue, grid, block, argsAveragePredictedValueKernel, numThreadsPerBlock * sizeof(int), stream[SECOND_STREAM_INDEX]));
    CUDA_CHECK(cudaEventRecord(secondStreamStop, stream[SECOND_STREAM_INDEX]));
    
    // On the device, wait for the recorded CUDA events from other streams to finish.
    CUDA_CHECK(cudaStreamWaitEvent(stream[SECOND_STREAM_INDEX], firstStreamStop));
    CUDA_CHECK(cudaStreamWaitEvent(stream[FIRST_STREAM_INDEX], secondStreamStop));
    
    // Second Step: launch the k_summationOfProducts kernel in the first stream.
    void *argsSumOfProductsKernel[] = {&observedValues_d, &predictedValues_d, &averageObservedValue_d, &averagePredictedValue_d, &summationOfProducts_d, &interimBufferSumOfProducts_d, &numElements};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_summationOfProducts, grid, block, argsSumOfProductsKernel, numThreadsPerBlock * sizeof(float), stream[FIRST_STREAM_INDEX]));
    
    // Second step: launch the k_summationOfSquares kernel in the second stream.
    void *argsSumOfSquaresKernel[] = {&observedValues_d, &averageObservedValue_d, &summationOfSquares_d, &interimBufferSumOfSquares_d, &numElements};
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_summationOfSquares, grid, block, argsSumOfSquaresKernel, numThreadsPerBlock * sizeof(float), stream[SECOND_STREAM_INDEX]));
    CUDA_CHECK(cudaEventRecord(secondStreamStop, stream[SECOND_STREAM_INDEX]));
    
    // On the device, wait for the recorded CUDA event from the other stream to finish. 
    CUDA_CHECK(cudaStreamWaitEvent(stream[FIRST_STREAM_INDEX], secondStreamStop));
    
    dim3 blockSize(LAST_KERNEL_NUMBER_OF_THREADS_PER_BLOCK, 1, 1);
    dim3 gridSize(LAST_KERNEL_NUMBER_OF_BLOCKS, 1, 1);
    
    // Grid: (1, 1, 1)
    // Block: (1, 1, 1)
    // Third step: launch the k_leastSquaresMethodResult kernel in the first stream.
    void *argsLeastSquaresMethodKernel[] = {&averageObservedValue_d, &averagePredictedValue_d, &summationOfProducts_d, &summationOfSquares_d, &outputSlope_d, &outputIntercept_d };
    CUDA_CHECK(cudaLaunchKernel((void*)k_leastSquaresMethodResult, gridSize, blockSize, argsLeastSquaresMethodKernel, 0, stream[FIRST_STREAM_INDEX]));
}