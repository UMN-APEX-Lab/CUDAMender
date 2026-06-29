#include "least_squares.h"
#include "least_squares_common.h"
#include <cooperative_groups.h>

using namespace cooperative_groups;

const int FIRST_STREAM_INDEX = 0;
const int SECOND_STREAM_INDEX = 1;
const int OUTPUT_SIZE = 1;

__global__ void k_computeAverageObservedValue(const int* observedValues_d, const int* predictedValues_d, float* averageObservedValue_d, int* interimBufferObservedVal_d, int inputSize) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    extern __shared__ int s_blockSummation[];
    
    int summation = 0;
    for(int threadIndex = threadId; threadIndex < inputSize; threadIndex += gridStride) {
        summation += observedValues_d[threadIndex];
    }
    
    s_blockSummation[threadIdx.x] = summation;
    __syncthreads();
    
    for(int offset = (blockDim.x >> 1); offset > 0; offset = (offset >> 1)) {
        if(threadIdx.x < offset) {
            s_blockSummation[threadIdx.x] += s_blockSummation[threadIdx.x + offset];
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) {
        interimBufferObservedVal_d[blockIdx.x] = s_blockSummation[threadIdx.x];
    }
    
    grid.sync();
    
    for(int offset = (gridDim.x >> 1); offset > 0; offset = (offset >> 1)) {
        if(threadIdx.x == 0) {
            if(blockIdx.x < offset) {
                interimBufferObservedVal_d[blockIdx.x] += interimBufferObservedVal_d[blockIdx.x + offset];
            }
        }
        grid.sync();
    }
    
    if(threadId == 0) {
        averageObservedValue_d[threadId] = (float)interimBufferObservedVal_d[threadId] / inputSize;
    }
}

__global__ void k_computeAveragePredictedValue(const int* observedValues_d, const int* predictedValues_d, float* averagePredictedValue_d, int* interimBufferPredictedVal_d, int inputSize) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    extern __shared__ int s_blockSummation[];
    
    int summation = 0;
    for(int threadIndex = threadId; threadIndex < inputSize; threadIndex += gridStride) {
        summation += predictedValues_d[threadIndex];
    }
    
    s_blockSummation[threadIdx.x] = summation;
    __syncthreads();
    
    for(int offset = (blockDim.x >> 1); offset > 0; offset = (offset >> 1)) {
        if(threadIdx.x < offset) {
            s_blockSummation[threadIdx.x] += s_blockSummation[threadIdx.x + offset];
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) {
        interimBufferPredictedVal_d[blockIdx.x] = s_blockSummation[threadIdx.x];
    }
    
    grid.sync();
    
    for(int offset = (gridDim.x >> 1); offset > 0; offset = (offset >> 1)) {
        if(threadIdx.x == 0) {
            if(blockIdx.x < offset) {
                interimBufferPredictedVal_d[blockIdx.x] += interimBufferPredictedVal_d[blockIdx.x + offset];
            }
        }
        grid.sync();
    }
    
    if(threadId == 0) {
        averagePredictedValue_d[threadId] = (float)interimBufferPredictedVal_d[threadId] / inputSize;
    }
}

__global__ void k_summationOfProducts(const int* observedValues_d, const int* predictedValues_d, float* averageObservedValue_d, float* averagePredictedValue_d, float* summationOfProducts_d, float* interimBufferSumOfProducts_d, int inputSize) {    
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    extern __shared__ float s_blockSumOfProducts[];
    
    float sumProducts = 0;
    for(int threadIndex = threadId; threadIndex < inputSize; threadIndex += gridStride) {
        float avgObservedVal = observedValues_d[threadIndex] - averageObservedValue_d[0];
        float avgPredictedVal = predictedValues_d[threadIndex] - averagePredictedValue_d[0];
        sumProducts += (avgObservedVal * avgPredictedVal);
    }
    
    s_blockSumOfProducts[threadIdx.x] = sumProducts;
    __syncthreads();
    
    for(int offset = (blockDim.x >> 1); offset > 0; offset = (offset >> 1)) {
        if(threadIdx.x < offset) {
            s_blockSumOfProducts[threadIdx.x] += s_blockSumOfProducts[threadIdx.x + offset];
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) {
        interimBufferSumOfProducts_d[blockIdx.x] = s_blockSumOfProducts[threadIdx.x];
    }
    
    grid.sync();
    
    for(int offset = (gridDim.x >> 1); offset > 0; offset = (offset >> 1)) {
        if(threadIdx.x == 0) {
            if(blockIdx.x < offset) {
                interimBufferSumOfProducts_d[blockIdx.x] += interimBufferSumOfProducts_d[blockIdx.x + offset];
            }
        }
        grid.sync();
    }
    
    if(threadId == 0) {
        summationOfProducts_d[threadId] = interimBufferSumOfProducts_d[threadId];
    }
}

__global__ void k_summationOfSquares(const int* observedValues_d, float* averageObservedValue_d, float* summationOfSquares_d, float* interimBufferSumOfSquares_d, int inputSize) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    extern __shared__ float s_blockSumOfSquares[];
    
    float sumSquares = 0;
    for(int threadIndex = threadId; threadIndex < inputSize; threadIndex += gridStride) {
        float avgObservedVal = observedValues_d[threadIndex] - averageObservedValue_d[0];
        sumSquares += (avgObservedVal * avgObservedVal);
    }
    
    s_blockSumOfSquares[threadIdx.x] = sumSquares;
    __syncthreads();
    
    for(int offset = (blockDim.x >> 1); offset > 0; offset = (offset >> 1)) {
        if(threadIdx.x < offset) {
            s_blockSumOfSquares[threadIdx.x] += s_blockSumOfSquares[threadIdx.x + offset];
        }
        __syncthreads();
    }
    
    if(threadIdx.x == 0) {
        interimBufferSumOfSquares_d[blockIdx.x] = s_blockSumOfSquares[threadIdx.x];
    }
    
    grid.sync();
    
    for(int offset = (gridDim.x >> 1); offset > 0; offset = (offset >> 1)) {
        if(threadIdx.x == 0) {
            if(blockIdx.x < offset) {
                interimBufferSumOfSquares_d[blockIdx.x] += interimBufferSumOfSquares_d[blockIdx.x + offset];
            }
        }
        grid.sync();
    }
    
    if(threadId == 0) {
        summationOfSquares_d[threadId] = interimBufferSumOfSquares_d[threadId];
    }
}

__global__ void k_leastSquaresMethodResult(float* averageObservedValue_d, float* averagePredictedValue_d, float* summationOfProducts_d, float* summationOfSquares_d, float* outputSlope_d, float* outputIntercept_d) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(threadId == 0) {
        if(summationOfSquares_d[threadId] == 0) {
            outputSlope_d[threadId] = nanf("NaN");
            outputIntercept_d[threadId] = averageObservedValue_d[threadId]; 
        } else {
            outputSlope_d[threadId] = summationOfProducts_d[threadId] / summationOfSquares_d[threadId];
            outputIntercept_d[threadId] = (averagePredictedValue_d[threadId] - (outputSlope_d[threadId] * averageObservedValue_d[threadId]));
        }
    }
}