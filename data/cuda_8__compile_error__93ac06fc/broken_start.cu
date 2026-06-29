#include "include/cuda_graph.h"
#include <cuda_runtime.h>

void run_cuda_graph(float* d_img, float* h_result, int width, int height) {
    // Device memory for intermediate results
    float* d_edge_result;
    float* d_normalized;
    float* d_blur_result;
    float* d_combined;
    float* d_final;
    
    int size = width * height * sizeof(float);
    
    // Allocate device memory
    cudaMalloc(&d_edge_result, size);
    cudaMalloc(&d_normalized, size);
    cudaMalloc(&d_blur_result, size);
    cudaMalloc(&d_combined, size);
    cudaMalloc(&d_final, size);
    
    // Create CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaGraphCreate(&graph, 0);
    
    // Block size
    int threads_per_block = 256;
    int total_threads = width * height;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    // Create kernel launch parameters for edge detection
    cudaKernelNodeParams edge_params = {0};
    void* edge_args[] = {(void*)&d_img, (void*)&d_edge_result, (void*)&width, (void*)&height};
    edge_params.func = (void*)apply_edge_detection;
    edge_params.gridDim = make_dim3(blocks, 1, 1);
    edge_params.blockDim = make_dim3(threads_per_block, 1, 1);
    edge_params.kernelParams = edge_args;
    edge_params.extra = NULL;
    
    cudaGraphNode_t edge_node;
    cudaGraphAddKernelNode(&edge_node, graph, NULL, 0, &edge_params);
    
    // Create kernel launch parameters for normalization
    cudaKernelNodeParams norm_params = {0};
    void* norm_args[] = {(void*)&d_edge_result, (void*)&width, (void*)&height};
    norm_params.func = (void*)normalize_image;
    norm_params.gridDim = make_dim3(blocks, 1, 1);
    norm_params.blockDim = make_dim3(threads_per_block, 1, 1);
    norm_params.kernelParams = norm_args;
    norm_params.extra = NULL;
    
    cudaGraphNode_t norm_node;
    cudaGraphAddKernelNode(&norm_node, graph, &edge_node, 1, &norm_params);
    
    // Create kernel launch parameters for blur filter
    cudaKernelNodeParams blur_params = {0};
    void* blur_args[] = {(void*)&d_normalized, (void*)&d_blur_result, (void*)&width, (void*)&height};
    blur_params.func = (void*)apply_blur_filter;
    blur_params.gridDim = make_dim3(blocks, 1, 1);
    blur_params.blockDim = make_dim3(threads_per_block, 1, 1);
    blur_params.kernelParams = blur_args;
    blur_params.extra = NULL;
    
    cudaGraphNode_t blur_node;
    cudaGraphAddKernelNode(&blur_node, graph, &norm_node, 1, &blur_params);
    
    // Create kernel launch parameters for combining results
    cudaKernelNodeParams combine_params = {0};
    void* combine_args[] = {(void*)&d_edge_result, (void*)&d_blur_result, (void*)&d_combined, (void*)&width, (void*)&height};
    combine_params.func = (void*)combine_filtered_results;
    combine_params.gridDim = make_dim3(blocks, 1, 1);
    combine_params.blockDim = make_dim3(threads_per_block, 1, 1);
    combine_params.kernelParams = combine_args;
    combine_params.extra = NULL;
    
    cudaGraphNode_t combine_node;
    cudaGraphAddKernelNode(&combine_node, graph, &blur_node, 1, &combine_params);
    
    // Create kernel launch parameters for final transformation
    cudaKernelNodeParams final_params = {0};
    void* final_args[] = {(void*)&d_combined, (void*)&width, (void*)&height};
    final_params.func = (void*)final_transformation;
    final_params.gridDim = make_dim3(blocks, 1, 1);
    final_params.blockDim = make_dim3(threads_per_block, 1, 1);
    final_params.kernelParams = final_args;
    final_params.extra = NULL;
    
    cudaGraphNode_t final_node;
    cudaGraphAddKernelNode(&final_node, graph, &combine_node, 1, &final_params);
    
    // Instantiate the graph
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    
    // Execute the graph 100 times
    for (int i = 0; i < 100; i++) {
        cudaGraphLaunch(graph_exec, 0);
    }
    
    // Copy final result back to host
    cudaMemcpy(h_result, d_final, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaFree(d_edge_result);
    cudaFree(d_normalized);
    cudaFree(d_blur_result);
    cudaFree(d_combined);
    cudaFree(d_final);
}