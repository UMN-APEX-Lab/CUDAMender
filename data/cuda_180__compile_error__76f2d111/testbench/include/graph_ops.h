#ifndef GRAPH_OPS_H
#define GRAPH_OPS_H

#include <cuda_runtime.h>

// Host function to create and run a cuda graph.
void runGraph(int k, float * dataIn_d, float * dataOut_d, int * constantParams_d, int maxActiveBlocks, cudaStream_t stream);

// CUDA kernel to do the calculations in CUDA graph.
__global__ void k_calculate(float * dataIn_d, float * dataOut_d, int * constantParams_d);

#endif // GRAPH_OPS_H