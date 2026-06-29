#include "include/least_squares.h"
#include <cuda_runtime.h>

void run(int numElements, int maxInputSize,
         cudaStream_t* stream, cudaEvent_t firstStreamStop, cudaEvent_t secondStreamStop,
         int* observedValues_d, int* predictedValues_d,
         int* interimBufferObservedVal_d, int* interimBufferPredictedVal_d,
         float* averageObservedValue_d, float* averagePredictedValue_d,
         float* summationOfProducts_d, float* summationOfSquares_d,
         float* interimBufferSumOfProducts_d, float* interimBufferSumOfSquares_d,
         float* outputSlope_d, float* outputIntercept_d)
{
    cudaStream_t s0 = stream[0];
    cudaStream_t s1 = stream[1];

    const dim3 grid(1);
    const dim3 block(maxInputSize);

    cudaMemsetAsync(interimBufferObservedVal_d, 0, maxInputSize * sizeof(int), s0);
    cudaMemsetAsync(averageObservedValue_d, 0, sizeof(float), s0);
    cudaMemsetAsync(summationOfProducts_d, 0, sizeof(float), s0);
    cudaMemsetAsync(interimBufferSumOfProducts_d, 0, maxInputSize * sizeof(float), s0);
    cudaMemsetAsync(outputSlope_d, 0, sizeof(float), s0);
    cudaMemsetAsync(outputIntercept_d, 0, sizeof(float), s0);

    cudaMemsetAsync(interimBufferPredictedVal_d, 0, maxInputSize * sizeof(int), s1);
    cudaMemsetAsync(averagePredictedValue_d, 0, sizeof(float), s1);
    cudaMemsetAsync(summationOfSquares_d, 0, sizeof(float), s1);
    cudaMemsetAsync(interimBufferSumOfSquares_d, 0, maxInputSize * sizeof(float), s1);

    k_computeAverageObservedValue<<<grid, block, 0, s0>>>(
        numElements,
        observedValues_d,
        interimBufferObservedVal_d,
        averageObservedValue_d);

    k_computeAveragePredictedValue<<<grid, block, 0, s1>>>(
        numElements,
        predictedValues_d,
        interimBufferPredictedVal_d,
        averagePredictedValue_d);

    cudaEventRecord(firstStreamStop, s0);
    cudaEventRecord(secondStreamStop, s1);

    cudaStreamWaitEvent(s0, secondStreamStop, 0);
    cudaStreamWaitEvent(s1, firstStreamStop, 0);

    k_summationOfProducts<<<grid, block, 0, s0>>>(
        numElements,
        observedValues_d,
        predictedValues_d,
        averageObservedValue_d,
        averagePredictedValue_d,
        interimBufferSumOfProducts_d,
        summationOfProducts_d);

    k_summationOfSquares<<<grid, block, 0, s1>>>(
        numElements,
        predictedValues_d,
        averagePredictedValue_d,
        interimBufferSumOfSquares_d,
        summationOfSquares_d);

    cudaEventRecord(secondStreamStop, s1);
    cudaStreamWaitEvent(s0, secondStreamStop, 0);

    k_leastSquaresMethodResult<<<1, 1, 0, s0>>>(
        averageObservedValue_d,
        averagePredictedValue_d,
        summationOfProducts_d,
        summationOfSquares_d,
        outputSlope_d,
        outputIntercept_d);
}