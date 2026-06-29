#include "graph_pipeline.h"

void runGraph(float* inpFrame, float* inpFrame_d, size_t frameHeight, size_t frameWidth, size_t numFrames, float gammaValue, float contrastValue, float brightnessValue, float* outRGB2Gray_d, float* outFrame_d, float* outFrame, cudaStream_t& stream) {
    const int DEFAULT_BLOCK_SIZE = 32;

    // Get GPU device properties
    cudaDeviceProp prop;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaGetDeviceProperties(&prop, device);

    size_t numElementsInFrame = frameHeight * frameWidth;
    float* inpFramePointer = inpFrame;
    float* outFramePointer = outFrame;

    // Block Size
    size_t block_size = DEFAULT_BLOCK_SIZE;
    size_t block_sizeX = fmin(block_size, prop.maxThreadsDim[0]);
    size_t block_sizeY = fmin(block_size, prop.maxThreadsDim[1]);
    dim3 blockDim(block_sizeX, block_sizeY, 1);

    // Grid-size 
    dim3 gridDim(ceil(frameWidth / (float)block_size), ceil(frameHeight / (float)block_size));
    gridDim.x = fmin(gridDim.x, prop.maxGridSize[0]);
    gridDim.y = fmin(gridDim.y, prop.maxGridSize[1]);

    // Create and initialize CUDA graph
    cudaGraph_t graph;
    CUDA_CHECK(cudaGraphCreate(&graph, 0));

    // Graphs Nodes and params
    cudaGraphNode_t memCpyNodeHtoD;
    cudaGraphNode_t memCpyNodeDtoH;
    cudaGraphNode_t rgbKernelNode;
    cudaGraphNode_t gammaCorrectionNode;

    cudaKernelNodeParams rgb2GrayKernelParams = { 0 };
    cudaKernelNodeParams gammaCorrectionKernelParams = { 0 };

    // Node initialization for RGB kernel
    void* rgbKernelArgs[] = { (void*)&inpFrame_d, (void*)&frameWidth, (void*)&frameHeight, (void*)&outRGB2Gray_d };
    rgb2GrayKernelParams.func = (void*)k_rgbToGray;
    rgb2GrayKernelParams.blockDim = blockDim;
    rgb2GrayKernelParams.gridDim = gridDim;
    rgb2GrayKernelParams.kernelParams = rgbKernelArgs;

    // Node initialization for image sharpening kernel
    void* gammaCorrectionKernelArgs[] = { (void*)&outRGB2Gray_d, (void*)&frameWidth, (void*)&frameHeight, (void*)&gammaValue, (void*)&contrastValue, (void*)&brightnessValue, (void*)&outFrame_d };
    gammaCorrectionKernelParams.func = (void*)k_gammaCorrection;
    gammaCorrectionKernelParams.blockDim = blockDim;
    gammaCorrectionKernelParams.gridDim = gridDim;
    gammaCorrectionKernelParams.kernelParams = gammaCorrectionKernelArgs;

    // Add nodes to graph
    CUDA_CHECK(cudaGraphAddMemcpyNode1D(&memCpyNodeHtoD, graph, NULL, 0, inpFrame_d, inpFramePointer, RGB_CHANNELS * numElementsInFrame * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaGraphAddKernelNode(&rgbKernelNode, graph, &memCpyNodeHtoD, 1, &rgb2GrayKernelParams));
    CUDA_CHECK(cudaGraphAddKernelNode(&gammaCorrectionNode, graph, &rgbKernelNode, 1, &gammaCorrectionKernelParams));
    CUDA_CHECK(cudaGraphAddMemcpyNode1D(&memCpyNodeDtoH, graph, &gammaCorrectionNode, 1, outFramePointer, outFrame_d, numElementsInFrame * sizeof(float), cudaMemcpyDeviceToHost));

    // Instantiate graph
    cudaGraphExec_t executableGraph;
    CUDA_CHECK(cudaGraphInstantiate(&executableGraph, graph));

    // Launch Graph and update input/output frame pointers
    int inputFrameOffset = numElementsInFrame * RGB_CHANNELS;
    int outputFrameOffset = numElementsInFrame;
    for (int frameIdx = 0;frameIdx < numFrames;frameIdx++) {
        // Update input frame pointer
        inpFramePointer = inpFrame + frameIdx * inputFrameOffset;
        CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(executableGraph, memCpyNodeHtoD, inpFrame_d, inpFramePointer, RGB_CHANNELS * numElementsInFrame * sizeof(float), cudaMemcpyHostToDevice));

        // Update output frame pointer
        outFramePointer = outFrame + frameIdx * outputFrameOffset;
        CUDA_CHECK(cudaGraphExecMemcpyNodeSetParams1D(executableGraph, memCpyNodeDtoH, outFramePointer, outFrame_d, numElementsInFrame * sizeof(float), cudaMemcpyDeviceToHost));

        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(executableGraph, stream));
    }

    // Destroy graph and execgraph instances
    CUDA_CHECK(cudaGraphExecDestroy(executableGraph));
    CUDA_CHECK(cudaGraphDestroy(graph));
}