#include "k_joinImages.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK(call)                                                           \
do {                                                                               \
        cudaError_t error = call;                                                  \
        if (error != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),\
                    __FILE__, __LINE__);                                           \
            exit(error);                                                           \
        }                                                                          \
} while (0)

#define TEST_CASES 10

#undef NDEBUG
#include <assert.h>

void launch() {

    int testCaseRows[TEST_CASES] = { 4,5,8,10,12,15,17,20,22,25 };
    int testCaseCols[TEST_CASES] = { 3,4,11,13,16,18,20,21,23,26 };

    for (int testcase = 0; testcase < TEST_CASES; testcase++) {

        // Initialization
        int numRows = testCaseRows[testcase];
        int numCols = testCaseCols[testcase];
        uchar* image1_h = (uchar*)malloc(numRows * numCols * sizeof(uchar));
        uchar* image2_h = (uchar*)malloc(numRows * numCols * sizeof(uchar));
        uchar* outputImage = (uchar*)malloc((2 * numCols) * numRows * sizeof(uchar));

        for (int i = 0; i < numRows * numCols; i++) {
            image1_h[i] = i + 1;
            image2_h[i] = i + 5;
        }

        // Running the code on CPU
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < numCols; col++) {
                outputImage[row * (2 * numCols) + col] = image1_h[row * numCols + col];
                outputImage[row * (2 * numCols) + col + numCols] = image2_h[row * numCols + col];
            }
        }

        // CUDA Initialization and memcpy
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        uchar* image1_d = nullptr; 
        uchar* image2_d = nullptr; 
        uchar* outputImage_d = nullptr;
        uchar* outputImage_h = (uchar*)malloc((2 * numCols) * numRows * sizeof(uchar));
        
        CUDA_CHECK(cudaMallocAsync(&image1_d, numRows * numCols * sizeof(uchar), stream));
        CUDA_CHECK(cudaMallocAsync(&image2_d, numRows * numCols * sizeof(uchar), stream));
        CUDA_CHECK(cudaMallocAsync(&outputImage_d, (2 * numCols) * numRows * sizeof(uchar), stream));

        CUDA_CHECK(cudaMemcpyAsync(image1_d, 
                                   image1_h, 
                                   numRows * numCols * sizeof(uchar), 
                                   cudaMemcpyHostToDevice, 
                                   stream));
        CUDA_CHECK(cudaMemcpyAsync(image2_d, 
                                   image2_h, 
                                   numRows * numCols * sizeof(uchar), 
                                   cudaMemcpyHostToDevice, 
                                   stream));

        // Running the code on GPU
        dim3 blockSize(2, 2);  // Each block will handle a 2x2 tile
        dim3 gridSize((numCols + blockSize.x - 1) / blockSize.x, (numRows + blockSize.y - 1) / blockSize.y);
        void* args[] = { &image1_d, &image2_d, &outputImage_d, (void*)&numCols, (void*)&numRows };
        CUDA_CHECK(cudaLaunchKernel((void*)k_joinImages, gridSize, blockSize, args, 0, stream));
        CUDA_CHECK(cudaMemcpyAsync(outputImage_h, 
                                   outputImage_d, 
                                   (2 * numCols) * numRows * sizeof(uchar), 
                                   cudaMemcpyDeviceToHost, 
                                   stream));

        // Verification
        for (int row = 0; row < numRows; row++) {
            for (int col = 0; col < 2 * numCols; col++) {
                assert(outputImage[row * (2 * numCols) + col] == outputImage_h[row * (2 * numCols) + col]);
            }
        }

        free(image1_h);
        free(image2_h);
        free(outputImage);
        free(outputImage_h);
        CUDA_CHECK(cudaFreeAsync(image1_d, stream));
        CUDA_CHECK(cudaFreeAsync(image2_d, stream));
        CUDA_CHECK(cudaFreeAsync(outputImage_d, stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

void benchmark() {
    const int width  = 16384;
    const int height = 16384;
    const size_t imgSize = (size_t)width * height * sizeof(uchar);
    const size_t outSize = (size_t)(2 * width) * height * sizeof(uchar);
    const int warmup = 3;
    const int timed  = 500;

    uchar* image1_h = (uchar*)malloc(imgSize);
    uchar* image2_h = (uchar*)malloc(imgSize);
    for (size_t i = 0; i < (size_t)width * height; i++) {
        image1_h[i] = (uchar)(i & 0xFF);
        image2_h[i] = (uchar)((i + 5) & 0xFF);
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    uchar *image1_d = nullptr, *image2_d = nullptr, *outputImage_d = nullptr;
    CUDA_CHECK(cudaMallocAsync(&image1_d, imgSize, stream));
    CUDA_CHECK(cudaMallocAsync(&image2_d, imgSize, stream));
    CUDA_CHECK(cudaMallocAsync(&outputImage_d, outSize, stream));

    CUDA_CHECK(cudaMemcpyAsync(image1_d, image1_h, imgSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(image2_d, image2_h, imgSize, cudaMemcpyHostToDevice, stream));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    void* args[] = { &image1_d, &image2_d, &outputImage_d, (void*)&width, (void*)&height };

    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_joinImages, gridSize, blockSize, args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_joinImages, gridSize, blockSize, args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(image1_d, stream));
    CUDA_CHECK(cudaFreeAsync(image2_d, stream));
    CUDA_CHECK(cudaFreeAsync(outputImage_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(image1_h);
    free(image2_h);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}