#include "graph_config.h"
#include "graph_ops.h"

// CUDA kernel to do the calculations in CUDA graph.
__global__ void k_calculate(float * dataIn_d, float * dataOut_d, int * constantParams_d) {
    int globalThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int numGlobalThreads = blockDim.x * gridDim.x;
    int matrixSize = constantParams_d[SELECT_PARAM_MATRIX_SIZE];
    int numIterationsForGridStrideLoop = (matrixSize * matrixSize + numGlobalThreads - 1) / numGlobalThreads;
    // Multiplying the matrix in dataIn_d with itself and writing the output to dataOut_d.
    for(int iteration = 0; iteration < numIterationsForGridStrideLoop; iteration++) {
        int workId = iteration * numGlobalThreads + globalThreadIndex;
        if(workId < matrixSize * matrixSize) {
            int outputColumn = workId % matrixSize;
            int outputRow = workId / matrixSize;
            float c = 0.0f;
            for(int i = 0; i < matrixSize; i++) {
                float a = dataIn_d[outputRow * matrixSize + i];
                float b = dataIn_d[outputColumn + i * matrixSize];
                c = fmaf(a, b, c);
            }
            dataOut_d[outputColumn + outputRow * matrixSize] = c;
        }
    }
}