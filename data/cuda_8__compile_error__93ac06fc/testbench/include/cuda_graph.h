#ifndef CUDA_GRAPH_H
#define CUDA_GRAPH_H

#include <cuda_runtime.h>

void run_cuda_graph(float* d_img, float* h_result, int width, int height);

#endif // CUDA_GRAPH_H