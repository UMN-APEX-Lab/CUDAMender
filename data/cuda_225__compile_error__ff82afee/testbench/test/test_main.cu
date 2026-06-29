#undef NDEBUG
#include <assert.h>
#include <limits.h>
#include <algorithm>
#include <cstring>
#include <nvtx3/nvToolsExt.h>
#include "least_squares.h"
#include "least_squares_common.h"

void launch() {
    constexpr int TEST_CASE_COUNT = 7;
    constexpr int MIN_ARRAY_SIZE = 1;
    constexpr int OUTPUT_INDEX = 0;
    
    constexpr int inputArraySize[TEST_CASE_COUNT] = { 9, 12, 16, 24, 32, 48, 64 };
    int maxInputSize = 0;
    for(int i = 0; i < TEST_CASE_COUNT; i++) {
        if(maxInputSize < inputArraySize[i]) {
            maxInputSize = inputArraySize[i];
        }
    }
    
    int * inputObservedValues_h = new int[TEST_CASE_COUNT * maxInputSize];
    int * inputPredictedValues_h = new int[TEST_CASE_COUNT * maxInputSize];
    int testIndex = 0;
    
    {
        std::initializer_list<int> initObservedValue = { 1, 8, 9, 6, 4, 3, 5, 7, 2 };
        std::initializer_list<int> initPredictedValue = { 8, 1, 7, 6, 2, 4, 5, 9, 3 };
        std::copy(initObservedValue.begin(), initObservedValue.end(), &inputObservedValues_h[maxInputSize * testIndex]);
        std::copy(initPredictedValue.begin(), initPredictedValue.end(), &inputPredictedValues_h[maxInputSize * testIndex]);
        testIndex++;
    }
    {
        std::initializer_list<int> initObservedValue = { 1, 2, 5, 15, 18, 16, 3, 14, 11, 19, 12, 13 };
        std::initializer_list<int> initPredictedValue = { 8, 2, 11, 1, 16, 5, 14, 9, 4, 3, 10, 7 };
        std::copy(initObservedValue.begin(), initObservedValue.end(), &inputObservedValues_h[maxInputSize * testIndex]);
        std::copy(initPredictedValue.begin(), initPredictedValue.end(), &inputPredictedValues_h[maxInputSize * testIndex]);
        testIndex++;
    }
    {
        std::initializer_list<int> initObservedValue = { 8, 3, 29, 27, 25, 7, 21, 15, 5, 19, 20, 28, 12, 16, 23, 24 };
        std::initializer_list<int> initPredictedValue = { 16, 9, 25, 27, 12, 8, 10, 5, 4, 23, 3, 13, 18, 21, 24, 29 };
        std::copy(initObservedValue.begin(), initObservedValue.end(), &inputObservedValues_h[maxInputSize * testIndex]);
        std::copy(initPredictedValue.begin(), initPredictedValue.end(), &inputPredictedValues_h[maxInputSize * testIndex]);
        testIndex++;
    }
    {
        std::initializer_list<int> initObservedValue = { 31, 12, 4, 29, 20, 22, 7, 9, 27, 1, 26, 16, 8, 2, 3, 14, 18, 5, 19, 15, 23, 13, 30, 11 };
        std::initializer_list<int> initPredictedValue = { 8, 30, 21, 17, 11, 2, 27, 5, 18, 7, 28, 19, 25, 12, 14, 26, 4, 3, 6, 15, 24, 23, 29, 1 };
        std::copy(initObservedValue.begin(), initObservedValue.end(), &inputObservedValues_h[maxInputSize * testIndex]);
        std::copy(initPredictedValue.begin(), initPredictedValue.end(), &inputPredictedValues_h[maxInputSize * testIndex]);
        testIndex++;
    }
    {
        std::initializer_list<int> initObservedValue = { 25, 7, 33, 12, 20, 32, 18, 22, 24, 19, 11, 36, 38, 31, 30, 17, 1, 14, 3, 16, 13, 34, 5, 21, 8, 4, 35, 27, 28, 37, 26, 9 };
        std::initializer_list<int> initPredictedValue = { 23, 18, 5, 16, 17, 33, 38, 9, 2, 1, 7, 26, 37, 24, 39, 31, 36, 27, 34, 12, 32, 25, 15, 10, 30, 21, 8, 22, 28, 35, 4, 29 };
        std::copy(initObservedValue.begin(), initObservedValue.end(), &inputObservedValues_h[maxInputSize * testIndex]);
        std::copy(initPredictedValue.begin(), initPredictedValue.end(), &inputPredictedValues_h[maxInputSize * testIndex]);
        testIndex++;
    }
    {
        std::initializer_list<int> initObservedValue = { 13, 5, 39, 51, 56, 16, 52, 20, 36, 25, 63, 10, 57, 41, 26, 33, 45, 32, 53, 4, 27, 40, 2, 21, 37, 34, 44, 17, 38, 12, 22, 55, 62, 1, 15, 42, 19, 54, 9, 60, 64, 7, 61, 49, 31, 58, 11, 28 };
        std::initializer_list<int> initPredictedValue = { 37, 30, 46, 14, 43, 61, 4, 42, 54, 24, 27, 1, 62, 2, 3, 64, 10, 52, 38, 57, 51, 33, 17, 28, 18, 49, 40, 41, 44, 22, 36, 9, 48, 26, 45, 6, 29, 5, 47, 20, 34, 32, 25, 58, 8, 60, 15, 12 };
        std::copy(initObservedValue.begin(), initObservedValue.end(), &inputObservedValues_h[maxInputSize * testIndex]);
        std::copy(initPredictedValue.begin(), initPredictedValue.end(), &inputPredictedValues_h[maxInputSize * testIndex]);
        testIndex++;
    }
    {
        std::initializer_list<int> initObservedValue = { 47, 77, 59, 16, 35, 11, 17, 29, 26, 20, 70, 2, 48, 30, 32, 6, 38, 24, 18, 63, 79, 41, 40, 1, 64, 22, 34, 53, 72, 46, 15, 76, 55, 13, 12, 37, 14, 73, 71, 7, 23, 8, 67, 27, 4, 50, 61, 78, 10, 31, 69, 52, 65, 74, 25, 60, 57, 66, 75, 33, 28, 56, 3, 62 };
        std::initializer_list<int> initPredictedValue = { 5, 50, 24, 37, 43, 30, 77, 78, 57, 73, 31, 36, 34, 2, 35, 76, 47, 22, 6, 48, 74, 70, 67, 56, 54, 7, 27, 51, 39, 4, 55, 19, 49, 68, 38, 41, 59, 8, 45, 69, 16, 66, 42, 75, 40, 18, 15, 14, 32, 25, 63, 72, 53, 62, 26, 10, 21, 33, 17, 13, 3, 20, 28, 29 };
        std::copy(initObservedValue.begin(), initObservedValue.end(), &inputObservedValues_h[maxInputSize * testIndex]);
        std::copy(initPredictedValue.begin(), initPredictedValue.end(), &inputPredictedValues_h[maxInputSize * testIndex]);
        testIndex++;
    }
    
    float expectedOutputSlope_h[TEST_CASE_COUNT] = {0.066, -0.083, 0.531, 0.145, -0.016, -0.010, -0.148};
    float expectedOutputIntercept_h[TEST_CASE_COUNT] = {4.666, 8.399, 6.078, 13.413, 22.024,  32.209, 45.089};
    
    float outputSlope_h[MIN_ARRAY_SIZE] = {};
    float outputIntercept_h[MIN_ARRAY_SIZE] = {};
    
    const int NUMBER_OF_STREAMS = 2;
    cudaStream_t stream[NUMBER_OF_STREAMS];
    for(int streamId = 0; streamId < NUMBER_OF_STREAMS; streamId++) {
        CUDA_CHECK(cudaStreamCreate(&stream[streamId]));
    }
    cudaEvent_t firstStreamStop;
    cudaEvent_t secondStreamStop;
    CUDA_CHECK(cudaEventCreate(&firstStreamStop));
    CUDA_CHECK(cudaEventCreate(&secondStreamStop));
    
    int* observedValues_d;
    int* predictedValues_d;
    int* interimBufferObservedVal_d;
    int* interimBufferPredictedVal_d;
    float* averageObservedValue_d;
    float* averagePredictedValue_d;
    float* summationOfProducts_d;
    float* summationOfSquares_d;
    float* interimBufferSumOfProducts_d;
    float* interimBufferSumOfSquares_d;
    float* outputSlope_d;
    float* outputIntercept_d;
    
    CUDA_CHECK(cudaMallocAsync((void**)&observedValues_d, maxInputSize * sizeof(int), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&predictedValues_d, maxInputSize * sizeof(int), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&interimBufferObservedVal_d, maxInputSize * sizeof(int), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&interimBufferPredictedVal_d, maxInputSize * sizeof(int), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&averageObservedValue_d, OUTPUT_SIZE * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&averagePredictedValue_d, OUTPUT_SIZE * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&summationOfProducts_d, OUTPUT_SIZE * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&summationOfSquares_d, OUTPUT_SIZE * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&interimBufferSumOfProducts_d, maxInputSize * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&interimBufferSumOfSquares_d, maxInputSize * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&outputSlope_d, OUTPUT_SIZE * sizeof(float), stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaMallocAsync((void**)&outputIntercept_d, OUTPUT_SIZE * sizeof(float), stream[FIRST_STREAM_INDEX]));
    
    for (int testCase = 0; testCase < TEST_CASE_COUNT; testCase++) {   
        int numElements = inputArraySize[testCase];
        
        CUDA_CHECK(cudaMemcpyAsync(observedValues_d, &inputObservedValues_h[testCase * maxInputSize], numElements * sizeof(int), cudaMemcpyHostToDevice, stream[FIRST_STREAM_INDEX]));
        CUDA_CHECK(cudaMemcpyAsync(predictedValues_d, &inputPredictedValues_h[testCase * maxInputSize], numElements * sizeof(int), cudaMemcpyHostToDevice, stream[FIRST_STREAM_INDEX]));
        
        run(numElements, maxInputSize, stream, firstStreamStop, secondStreamStop, observedValues_d, predictedValues_d, interimBufferObservedVal_d, interimBufferPredictedVal_d, averageObservedValue_d, averagePredictedValue_d, summationOfProducts_d, summationOfSquares_d, interimBufferSumOfProducts_d, interimBufferSumOfSquares_d, outputSlope_d, outputIntercept_d);
        
        CUDA_CHECK(cudaMemcpyAsync(outputSlope_h, outputSlope_d, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream[FIRST_STREAM_INDEX]));
        CUDA_CHECK(cudaMemcpyAsync(outputIntercept_h, outputIntercept_d, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream[FIRST_STREAM_INDEX]));
        CUDA_CHECK(cudaStreamSynchronize(stream[FIRST_STREAM_INDEX]));
        
        assert(fabs(outputIntercept_h[OUTPUT_INDEX] - expectedOutputIntercept_h[testCase]) <= EPSILON);
        isnan(expectedOutputSlope_h[OUTPUT_INDEX]) ? assert(isnan(outputSlope_h[OUTPUT_INDEX])) : assert(fabs(outputSlope_h[OUTPUT_INDEX] - expectedOutputSlope_h[testCase]) <= EPSILON);
    }
    delete [] inputObservedValues_h;
    delete [] inputPredictedValues_h;
    
    CUDA_CHECK(cudaFreeAsync(observedValues_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(predictedValues_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(interimBufferObservedVal_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(interimBufferPredictedVal_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(averageObservedValue_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(averagePredictedValue_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(summationOfProducts_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(interimBufferSumOfProducts_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(interimBufferSumOfSquares_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(summationOfSquares_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(outputSlope_d, stream[FIRST_STREAM_INDEX]));
    CUDA_CHECK(cudaFreeAsync(outputIntercept_d, stream[FIRST_STREAM_INDEX]));
    
    CUDA_CHECK(cudaEventDestroy(firstStreamStop));
    CUDA_CHECK(cudaEventDestroy(secondStreamStop));
    
    for(int streamId = 0; streamId < NUMBER_OF_STREAMS; streamId++) {
        CUDA_CHECK(cudaStreamSynchronize(stream[streamId]));
        CUDA_CHECK(cudaStreamDestroy(stream[streamId]));
    }
}

void benchmark() {
    constexpr int PERF_NUM_ELEMENTS = 65536;
    constexpr int PERF_MAX_INPUT_SIZE = PERF_NUM_ELEMENTS;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 100;

    const int NUMBER_OF_STREAMS = 2;
    cudaStream_t stream[NUMBER_OF_STREAMS];
    for (int i = 0; i < NUMBER_OF_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }
    cudaEvent_t firstStreamStop, secondStreamStop;
    CUDA_CHECK(cudaEventCreate(&firstStreamStop));
    CUDA_CHECK(cudaEventCreate(&secondStreamStop));

    int* observedValues_h = new int[PERF_NUM_ELEMENTS];
    int* predictedValues_h = new int[PERF_NUM_ELEMENTS];
    for (int i = 0; i < PERF_NUM_ELEMENTS; i++) {
        observedValues_h[i] = (i * 37 + 13) % 10000;
        predictedValues_h[i] = (i * 53 + 7) % 10000;
    }

    int* observedValues_d;
    int* predictedValues_d;
    int* interimBufferObservedVal_d;
    int* interimBufferPredictedVal_d;
    float* averageObservedValue_d;
    float* averagePredictedValue_d;
    float* summationOfProducts_d;
    float* summationOfSquares_d;
    float* interimBufferSumOfProducts_d;
    float* interimBufferSumOfSquares_d;
    float* outputSlope_d;
    float* outputIntercept_d;

    CUDA_CHECK(cudaMalloc((void**)&observedValues_d, PERF_MAX_INPUT_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&predictedValues_d, PERF_MAX_INPUT_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&interimBufferObservedVal_d, PERF_MAX_INPUT_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&interimBufferPredictedVal_d, PERF_MAX_INPUT_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&averageObservedValue_d, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&averagePredictedValue_d, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&summationOfProducts_d, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&summationOfSquares_d, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&interimBufferSumOfProducts_d, PERF_MAX_INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&interimBufferSumOfSquares_d, PERF_MAX_INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&outputSlope_d, OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&outputIntercept_d, OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(observedValues_d, observedValues_h, PERF_NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(predictedValues_d, predictedValues_h, PERF_NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < WARMUP_ITERS; i++) {
        run(PERF_NUM_ELEMENTS, PERF_MAX_INPUT_SIZE, stream, firstStreamStop, secondStreamStop,
            observedValues_d, predictedValues_d,
            interimBufferObservedVal_d, interimBufferPredictedVal_d,
            averageObservedValue_d, averagePredictedValue_d,
            summationOfProducts_d, summationOfSquares_d,
            interimBufferSumOfProducts_d, interimBufferSumOfSquares_d,
            outputSlope_d, outputIntercept_d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        run(PERF_NUM_ELEMENTS, PERF_MAX_INPUT_SIZE, stream, firstStreamStop, secondStreamStop,
            observedValues_d, predictedValues_d,
            interimBufferObservedVal_d, interimBufferPredictedVal_d,
            averageObservedValue_d, averagePredictedValue_d,
            summationOfProducts_d, summationOfSquares_d,
            interimBufferSumOfProducts_d, interimBufferSumOfSquares_d,
            outputSlope_d, outputIntercept_d);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    CUDA_CHECK(cudaFree(observedValues_d));
    CUDA_CHECK(cudaFree(predictedValues_d));
    CUDA_CHECK(cudaFree(interimBufferObservedVal_d));
    CUDA_CHECK(cudaFree(interimBufferPredictedVal_d));
    CUDA_CHECK(cudaFree(averageObservedValue_d));
    CUDA_CHECK(cudaFree(averagePredictedValue_d));
    CUDA_CHECK(cudaFree(summationOfProducts_d));
    CUDA_CHECK(cudaFree(summationOfSquares_d));
    CUDA_CHECK(cudaFree(interimBufferSumOfProducts_d));
    CUDA_CHECK(cudaFree(interimBufferSumOfSquares_d));
    CUDA_CHECK(cudaFree(outputSlope_d));
    CUDA_CHECK(cudaFree(outputIntercept_d));

    CUDA_CHECK(cudaEventDestroy(firstStreamStop));
    CUDA_CHECK(cudaEventDestroy(secondStreamStop));
    for (int i = 0; i < NUMBER_OF_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(stream[i]));
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }

    delete[] observedValues_h;
    delete[] predictedValues_h;
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}