#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "image_erosion.h"
#include "erosion_constants.h"

#define CUDA_CHECK(call)                                        \
do {                                                            \
        cudaError_t error = call;                               \
        if (error != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",       \
                    __FILE__, __LINE__,                         \
                    cudaGetErrorString(error));                 \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
} while(0)

void launch() {

    //Initialize Constants
    const int TEST_CASE_COUNT = 7;
    const int MAX_INPUT_IMAGE_WIDTH = 9;
    const int MAX_INPUT_IMAGE_HEIGHT = 9;
    const int MAX_IMAGE_DIMENSIONS = 2;
    const int IMAGE_HEIGHT_INDEX = 0;
    const int IMAGE_WIDTH_INDEX = 1;
    const int MIN_NUMBER_OF_THREADS_PER_BLOCK = 32;
    const int MAX_NUMBER_OF_BLOCKS = 4;
    
    //Use CUDA Streams for Asynchronous Execution
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    //Allocate Device Memory
    unsigned char *inputImage_d;
    unsigned char *structuringElement_d;
    unsigned char *outputImage_d;
    unsigned char *outputImageIntermediateBuffer_d;
    
    CUDA_CHECK(cudaMallocAsync((void**)&inputImage_d, MAX_INPUT_IMAGE_WIDTH * MAX_INPUT_IMAGE_HEIGHT * sizeof(unsigned char), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&structuringElement_d, MAX_INPUT_IMAGE_WIDTH * MAX_INPUT_IMAGE_HEIGHT * sizeof(unsigned char), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&outputImage_d, MAX_INPUT_IMAGE_WIDTH * MAX_INPUT_IMAGE_HEIGHT * sizeof(unsigned char), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&outputImageIntermediateBuffer_d, MAX_INPUT_IMAGE_WIDTH * MAX_INPUT_IMAGE_HEIGHT * sizeof(unsigned char), stream));
   
    //Initialise Test Data
    //Test Data Dimensions
    int inputImageWidthHeight[TEST_CASE_COUNT][MAX_IMAGE_DIMENSIONS] = {
      //Test Case - 1, {rows(height), columns(width)} 
      {4, 5},
      //Test Case - 2
      {5, 6},
      //Test Case - 3
      {6, 7},
      //Test Case - 4
      {7, 8},
      //Test Case - 5
      {8, 8},
      //Test Case - 6
      {9, 7},
      //Test Case - 7
      {9, 9}
    };

    int structuringElementWidthHeight[MAX_IMAGE_DIMENSIONS] = {3, 3};

    //Input Data For Test
    unsigned char inputImage_h[TEST_CASE_COUNT][MAX_INPUT_IMAGE_WIDTH * MAX_INPUT_IMAGE_HEIGHT] = {
      //Test Case - 1
      {1, 1, 1, 1, 1,
       1, 1, 1, 1, 1,
       1, 1, 1, 1, 1,
       1, 1, 1, 1, 1},
      //Test Case - 2
      {0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 0, 0},
      //Test Case - 3 
      {0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 0, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1},
      //Test Case - 4 
      {1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1,
       0, 1, 1, 1, 1, 1, 0, 1},
      //Test Case - 5 
      {1, 1, 1, 1, 1, 1, 1, 0,
       1, 1, 1, 1, 1, 1, 0, 1,
       1, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 0,
       0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 0, 0, 1, 1, 0, 1},
      //Test Case - 6 
      {1, 1, 1, 1, 1, 1, 1,
       1, 0, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1,
       1, 0, 1, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1},
      //Test Case - 7 
      {1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 0,
       1, 1, 1, 1, 0, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 0, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 0, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1}
    };

    //Expected Output for Test
    unsigned char expectedOutputImage_h[TEST_CASE_COUNT][MAX_INPUT_IMAGE_WIDTH * MAX_INPUT_IMAGE_HEIGHT] = {
      //Test Case - 1 
      {1, 1, 1, 1, 1,
       1, 1, 1, 1, 1,
       1, 1, 1, 1, 1,
       1, 1, 1, 1, 1},
      //Test Case - 2 
      {0, 0, 0, 1, 1, 1, 
       0, 0, 0, 1, 1, 1, 
       0, 0, 0, 0, 0, 0, 
       1, 1, 0, 0, 0, 0,
       1, 1, 0, 0, 0, 0},
      //Test Case - 3 
      {0, 0, 0, 1, 1, 1, 1, 
       0, 0, 0, 0, 0, 0, 1, 
       0, 0, 0, 0, 0, 0, 1, 
       1, 0, 0, 0, 0, 0, 1, 
       1, 0, 0, 0, 0, 0, 1, 
       1, 0, 0, 0, 0, 0, 1},
      //Test Case - 4 
      {1, 1, 1, 1, 1, 1, 1, 1, 
       1, 1, 1, 1, 1, 1, 1, 1, 
       1, 1, 1, 1, 1, 1, 1, 1, 
       1, 1, 1, 1, 1, 1, 1, 1, 
       0, 0, 0, 1, 0, 0, 0, 0, 
       0, 0, 0, 1, 0, 0, 0, 0, 
       0, 0, 0, 1, 0, 0, 0, 0},
      //Test Case - 5 
      {1, 0, 0, 0, 0, 0, 0, 0, 
       1, 0, 0, 0, 0, 0, 0, 0, 
       1, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0},
      //Test Case - 6 
      {0, 0, 0, 0, 1, 1, 1, 
       0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 
       1, 1, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0},
      //Test Case - 7
      {1, 0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 0, 
       0, 0, 0, 0, 0, 0, 0, 0, 1, 
       0, 0, 0, 0, 0, 0, 1, 1, 1, 
       0, 0, 0, 0, 0, 0, 1, 1, 1, 
       0, 0, 0, 0, 0, 0, 1, 1, 1} 
    };

    //Structuring Element
    unsigned char structuringElement_h[MAX_INPUT_IMAGE_HEIGHT * MAX_INPUT_IMAGE_WIDTH] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    
    //Erosion Iterations
    int erosionIterations[TEST_CASE_COUNT] = {1, 2, 2, 2, 2, 2, 3}; 

    //Output Image
    unsigned char outputImage_h[MAX_INPUT_IMAGE_WIDTH * MAX_INPUT_IMAGE_HEIGHT];

    //Execute Test Cases
    for (int testCase = 0; testCase < TEST_CASE_COUNT; testCase++){
      int inputImageHeight = inputImageWidthHeight[testCase][IMAGE_HEIGHT_INDEX];  
      int inputImageWidth = inputImageWidthHeight[testCase][IMAGE_WIDTH_INDEX];
      int structuringElementHeight = structuringElementWidthHeight[IMAGE_HEIGHT_INDEX];
      int structuringElementWidth = structuringElementWidthHeight[IMAGE_WIDTH_INDEX];
      int numberOfErosionIterations = erosionIterations[testCase];
      
      //copy data from host to device
      CUDA_CHECK(cudaMemcpyAsync(inputImage_d, inputImage_h[testCase], inputImageWidth * inputImageHeight * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(structuringElement_d, structuringElement_h, structuringElementWidth * structuringElementHeight * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));
      
      //Set Kernel Configuration
      int numThreadsPerBlock = MIN_NUMBER_OF_THREADS_PER_BLOCK;
      if( ceil((float)(inputImageWidth * inputImageHeight) / numThreadsPerBlock) > MAX_NUMBER_OF_BLOCKS){
          numThreadsPerBlock = ceil((float)(inputImageWidth * inputImageHeight) / MAX_NUMBER_OF_BLOCKS) ;
      }

      int numBlocks = ceil((float)(inputImageWidth * inputImageHeight) / numThreadsPerBlock);
      dim3 block(numThreadsPerBlock, 1, 1);
      dim3 grid(numBlocks, 1, 1);
      
      //Launch Kernel
      // Grid:  ((inputImageWidth * inputImageHeight) / numThreadsPerBlock, 1, 1)
      // Block: (32, 1, 1)
      void *args[] = {&inputImage_d, &structuringElement_d, &outputImage_d, &outputImageIntermediateBuffer_d, &inputImageWidth, &inputImageHeight, &structuringElementWidth, &structuringElementHeight, &numberOfErosionIterations};
      CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_imageErosion, grid, block, args, sizeof(unsigned char), stream));
      
      //Copy Data from device to host
      CUDA_CHECK(cudaMemcpyAsync(outputImage_h, outputImage_d, inputImageWidth * inputImageHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
      
      //Sycnhronize tasks in the stream
      CUDA_CHECK(cudaStreamSynchronize(stream));
      
      //Assert device output and expected output
      for(int rowIndex = MIN_IMAGE_ROW_INDEX; rowIndex < inputImageHeight; rowIndex++) {
        for(int columnIndex = MIN_IMAGE_COLUMN_INDEX; columnIndex < inputImageWidth; columnIndex++) {
            int pixelIndex = rowIndex * inputImageWidth + columnIndex;
            assert(outputImage_h[pixelIndex] == expectedOutputImage_h[testCase][pixelIndex]);
        }
      }
    }
    
    //Deallocate Device Memory
    CUDA_CHECK(cudaFreeAsync(inputImage_d, stream));
    CUDA_CHECK(cudaFreeAsync(structuringElement_d, stream));
    CUDA_CHECK(cudaFreeAsync(outputImage_d, stream));
    CUDA_CHECK(cudaFreeAsync(outputImageIntermediateBuffer_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    const int IMG_W = 512;
    const int IMG_H = 512;
    const int IMG_SIZE = IMG_W * IMG_H;
    const int SE_W = 3;
    const int SE_H = 3;
    const int EROSION_ITERS = 5;
    const int WARMUP = 3;
    const int TIMED = 100;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    unsigned char *inputImage_d, *structuringElement_d, *outputImage_d, *intermediateBuf_d;
    CUDA_CHECK(cudaMalloc(&inputImage_d, IMG_SIZE));
    CUDA_CHECK(cudaMalloc(&structuringElement_d, SE_W * SE_H));
    CUDA_CHECK(cudaMalloc(&outputImage_d, IMG_SIZE));
    CUDA_CHECK(cudaMalloc(&intermediateBuf_d, IMG_SIZE));

    unsigned char *inputImage_h = (unsigned char*)malloc(IMG_SIZE);
    unsigned char se_h[SE_W * SE_H];
    for (int i = 0; i < IMG_SIZE; i++) inputImage_h[i] = (i % 7 == 0) ? 0 : 1;
    for (int i = 0; i < SE_W * SE_H; i++) se_h[i] = 1;

    CUDA_CHECK(cudaMemcpy(inputImage_d, inputImage_h, IMG_SIZE, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(structuringElement_d, se_h, SE_W * SE_H, cudaMemcpyHostToDevice));

    int numThreadsPerBlock = 256;
    int maxBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, (void*)k_imageErosion, numThreadsPerBlock, sizeof(unsigned char)));

    int smCount = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
    int totalMaxBlocks = maxBlocks * smCount;

    int neededBlocks = (IMG_SIZE + numThreadsPerBlock - 1) / numThreadsPerBlock;
    if (neededBlocks > totalMaxBlocks) neededBlocks = totalMaxBlocks;

    dim3 block(numThreadsPerBlock, 1, 1);
    dim3 grid(neededBlocks, 1, 1);

    int w = IMG_W, h = IMG_H, sw = SE_W, sh = SE_H, iters = EROSION_ITERS;
    void *args[] = {&inputImage_d, &structuringElement_d, &outputImage_d,
                    &intermediateBuf_d, &w, &h, &sw, &sh, &iters};

    for (int i = 0; i < WARMUP; i++) {
        CUDA_CHECK(cudaMemcpy(inputImage_d, inputImage_h, IMG_SIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_imageErosion, grid, block, args, sizeof(unsigned char), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED; i++) {
        CUDA_CHECK(cudaMemcpy(inputImage_d, inputImage_h, IMG_SIZE, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_imageErosion, grid, block, args, sizeof(unsigned char), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    CUDA_CHECK(cudaFree(inputImage_d));
    CUDA_CHECK(cudaFree(structuringElement_d));
    CUDA_CHECK(cudaFree(outputImage_d));
    CUDA_CHECK(cudaFree(intermediateBuf_d));
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(inputImage_h);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}