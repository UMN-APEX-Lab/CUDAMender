#include "graph_config.h"
#include "graph_ops.h"

void runGraph(int k, float * dataIn_d, float * dataOut_d, int * constantParams_d, int maxActiveBlocks, cudaStream_t stream) {

    // Creating a graph with two nodes.
    cudaGraph_t graph;
    cudaGraphExec_t executableGraph;
    cudaGraphNode_t nodes[NUM_NODES];
    cudaKernelNodeParams nodeParams;
    
    CUDA_CHECK(cudaGraphCreate(&graph, 0));
    
    // Topology of the graph utilized:
    //
    //        kernel node: k_calculate
    //           |
    //           V
    //        memcpy node: device to device (dataIn_d <- dataOut_d) as an in-place feedback loop.
    
    // Adding nodes.
    {
        // Creating a node for the matrix-square kernel.
        int requiredBlocks = (MAX_MATRIX_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int usedBlocks = maxActiveBlocks < requiredBlocks ? maxActiveBlocks : requiredBlocks;
        // Grid: (usedBlocks, 1, 1)
        // Block: (BLOCK_SIZE, 1, 1)
        nodeParams = {};
        nodeParams.func = (void*)k_calculate;
        nodeParams.gridDim = dim3(usedBlocks, 1, 1);
        nodeParams.blockDim = dim3(BLOCK_SIZE, 1, 1);
        nodeParams.kernelParams = new void*[3];
        nodeParams.kernelParams[0] = (void*)&dataIn_d;
        nodeParams.kernelParams[1] = (void*)&dataOut_d;
        nodeParams.kernelParams[2] = (void*)&constantParams_d;
        nodeParams.sharedMemBytes = 0;
        nodeParams.extra = nullptr;
        CUDA_CHECK(cudaGraphAddKernelNode(  &nodes[NODE_KERNEL], 
                                            graph, 
                                            HAS_NO_DEPENDENCY, 
                                            ZERO_NODES_AS_DEPENDENCY, 
                                            &nodeParams));
        // Creating memcpy node for the in-place feedback loop.
        CUDA_CHECK(cudaGraphAddMemcpyNode1D(&nodes[NODE_MEMCPY_DEVICE_TO_DEVICE], graph, &nodes[NODE_KERNEL], ONE_NODE_AS_DEPENDENCY, dataIn_d, dataOut_d, sizeof(float) * MAX_MATRIX_ELEMENTS, cudaMemcpyDeviceToDevice));
        // Instantiating the graph.
        CUDA_CHECK(cudaGraphInstantiate(&executableGraph, graph));
    }

    // Executing graph to calculate matrix ^ (2 ^ k).
    for(int i = 0; i < k; i++) {
        CUDA_CHECK(cudaGraphLaunch(executableGraph, stream));
    }
    // Synchronizing before deallocating the graph.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Deallocating the graph.
    delete [] nodeParams.kernelParams;
    CUDA_CHECK(cudaGraphExecDestroy(executableGraph));
    CUDA_CHECK(cudaGraphDestroy(graph));
}