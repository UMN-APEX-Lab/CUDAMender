#include "cuda_graph.h"
#include "kernels.h"

void run_cuda_graph(float* d_img, float* h_result, int width, int height)
{
    const int size  = width * height;
    const int bytes = size * sizeof(float);
    float* d_edge_result;
    float* d_blur_result;
    float* d_combined_result;

    cudaMalloc(&d_edge_result, bytes);
    cudaMalloc(&d_blur_result, bytes);
    cudaMalloc(&d_combined_result, bytes);

    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);

    // Create the CUDA graph
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphNode_t edgeNode, normalizeNode, blurNode, combineNode, finalNode;

    // Set up kernel node parameters
    void* edgeArgs[] = {(void*)&d_img, (void*)&d_edge_result, (void*)&width, (void*)&height};
    cudaKernelNodeParams edgeParams = {0};
    edgeParams.func                 = (void*)apply_edge_detection;
    edgeParams.gridDim              = grid;
    edgeParams.blockDim             = block;
    edgeParams.sharedMemBytes       = 0;
    edgeParams.kernelParams         = edgeArgs;
    edgeParams.extra                = NULL;

    void* normalizeArgs[]                = {(void*)&d_edge_result, (void*)&width, (void*)&height};
    cudaKernelNodeParams normalizeParams = {0};
    normalizeParams.func                 = (void*)normalize_image;
    normalizeParams.gridDim              = grid;
    normalizeParams.blockDim             = block;
    normalizeParams.sharedMemBytes       = 0;
    normalizeParams.kernelParams         = normalizeArgs;
    normalizeParams.extra                = NULL;

    void* blurArgs[]                = {(void*)&d_edge_result, (void*)&d_blur_result, (void*)&width,
                                       (void*)&height};
    cudaKernelNodeParams blurParams = {0};
    blurParams.func                 = (void*)apply_blur_filter;
    blurParams.gridDim              = grid;
    blurParams.blockDim             = block;
    blurParams.sharedMemBytes       = 0;
    blurParams.kernelParams         = blurArgs;
    blurParams.extra                = NULL;

    void* combineArgs[] = {(void*)&d_edge_result, (void*)&d_blur_result, (void*)&d_combined_result,
                           (void*)&width, (void*)&height};
    cudaKernelNodeParams combineParams = {0};
    combineParams.func                 = (void*)combine_filtered_results;
    combineParams.gridDim              = grid;
    combineParams.blockDim             = block;
    combineParams.sharedMemBytes       = 0;
    combineParams.kernelParams         = combineArgs;
    combineParams.extra                = NULL;

    void* finalArgs[]                = {(void*)&d_combined_result, (void*)&width, (void*)&height};
    cudaKernelNodeParams finalParams = {0};
    finalParams.func                 = (void*)final_transformation;
    finalParams.gridDim              = grid;
    finalParams.blockDim             = block;
    finalParams.sharedMemBytes       = 0;
    finalParams.kernelParams         = finalArgs;
    finalParams.extra                = NULL;

    // Create the graph
    cudaGraphCreate(&graph, 0);

    cudaGraphAddKernelNode(&edgeNode, graph, NULL, 0, &edgeParams);
    cudaGraphAddKernelNode(&normalizeNode, graph, &edgeNode, 1, &normalizeParams);
    cudaGraphAddKernelNode(&blurNode, graph, &normalizeNode, 1, &blurParams);
    cudaGraphNode_t combineDependencies[] = {blurNode, normalizeNode};
    cudaGraphAddKernelNode(&combineNode, graph, combineDependencies, 2, &combineParams);
    cudaGraphAddKernelNode(&finalNode, graph, &combineNode, 1, &finalParams);

    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    // Execute the graph 100 times
    for (int i = 0; i < 100; ++i)
    {
        cudaGraphLaunch(instance, stream);
    }

    // Copy the final result to host
    cudaMemcpy(h_result, d_combined_result, bytes, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaStreamSynchronize(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);

    cudaFree(d_edge_result);
    cudaFree(d_blur_result);
    cudaFree(d_combined_result);
}