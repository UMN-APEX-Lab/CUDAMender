#ifndef LEAST_SQUARES_H
#define LEAST_SQUARES_H

#include <cuda_runtime.h>

void run(int numElements, int maxInputSize, 
         cudaStream_t* stream, cudaEvent_t firstStreamStop, cudaEvent_t secondStreamStop, 
         int* observedValues_d, int* predictedValues_d, 
         int* interimBufferObservedVal_d, int* interimBufferPredictedVal_d, 
         float* averageObservedValue_d, float* averagePredictedValue_d, 
         float* summationOfProducts_d, float* summationOfSquares_d, 
         float* interimBufferSumOfProducts_d, float* interimBufferSumOfSquares_d, 
         float* outputSlope_d, float* outputIntercept_d);

__global__ void k_computeAverageObservedValue(const int* observedValues_d, const int* predictedValues_d, 
                                               float* averageObservedValue_d, int* interimBufferObservedVal_d, 
                                               int inputSize);

__global__ void k_computeAveragePredictedValue(const int* observedValues_d, const int* predictedValues_d, 
                                                float* averagePredictedValue_d, int* interimBufferPredictedVal_d, 
                                                int inputSize);

__global__ void k_summationOfProducts(const int* observedValues_d, const int* predictedValues_d, 
                                       float* averageObservedValue_d, float* averagePredictedValue_d, 
                                       float* summationOfProducts_d, float* interimBufferSumOfProducts_d, 
                                       int inputSize);

__global__ void k_summationOfSquares(const int* observedValues_d, float* averageObservedValue_d, 
                                      float* summationOfSquares_d, float* interimBufferSumOfSquares_d, 
                                      int inputSize);

__global__ void k_leastSquaresMethodResult(float* averageObservedValue_d, float* averagePredictedValue_d, 
                                            float* summationOfProducts_d, float* summationOfSquares_d, 
                                            float* outputSlope_d, float* outputIntercept_d);

#endif // LEAST_SQUARES_H