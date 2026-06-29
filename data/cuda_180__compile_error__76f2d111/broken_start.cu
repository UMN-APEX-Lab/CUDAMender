#include <cuda_runtime.h>
#include "graph_ops.h"
#include "graph_config.h"

void runGraph(int k, float *dataIn_d, float *dataOut_d, int *constantParams_d, int maxActiveBlocks, cudaStream_t stream) {
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphNode_t kernelNode, copyNode;

    // Create the graph
    cudaGraphCreate(&graph, 0);

    // Kernel configuration
    int matrixSize = 0;
    cudaMemcpyFromDevice(&matrixSize, constantParams_d, sizeof(int), stream);
    
    // In a real scenario, we should wait for the copy to finish or use a host-side value.
    // However, for the graph construction, we need the grid/block dimensions.
    // Assuming the matrix size is accessible or provided in constantParams_d.
    // Since we can't easily block on the stream here without sync, we assume 
    // the matrixSize is handled by the kernel internal logic or we sync.
    cudaStreamSynchronize(stream);
    cudaMemcpyFromDevice(&matrixSize, constantParams_d, sizeof(int), nullptr);

    int threadsPerBlock = 256;
    int blocksPerGrid = (matrixSize + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid > maxActiveBlocks) blocksPerGrid = maxActiveBlocks;

    void* kernelArgs[] = { &dataIn_d, &dataOut_d, &constantParams_d };
    cudaKernelNodeParams kernelParams = {0};
    kernelParams.func = (void*)k_calculate;
    kernelParams.gridDim = dim3(blocksPerGrid, 1, 1);
    kernelParams.blockDim = dim3(threadsPerBlock, 1, 1);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;

    // Add kernel node
    cudaGraphAddKernelNode(&kernelNode, graph, &kernelParams);

    // Add D2D copy node (feedback loop: dataOut_d -> dataIn_d)
    cudaGraphNodeParams copyParams = {0};
    copyParams.type = CudaGraphNodeMemcpy;
    copyParams.srcDevPtr = dataOut_d;
    copyParams.dstDevPtr = dataIn_d;
    copyParams.sizeBytes = matrixSize * sizeof(float);

    cudaGraphAddMemcpyNode(&copyNode, graph, &copyParams);

    // Define dependencies: Kernel -> Copy
    cudaGraphNode_t dependencies[] = { kernelNode };
    cudaGraphNodeSetDependencies(copyNode, 1, dependencies);

    // Instantiate and execute the graph k times
    cudaGraphInstantiate(&instance, graph, &stream, NULL, 0, 0);

    for (int i = 0; i < k; ++i) {
        cudaGraphLaunch(instance, stream);
    }

    cudaStreamSynchronize(stream);

    // Cleanup
    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
}