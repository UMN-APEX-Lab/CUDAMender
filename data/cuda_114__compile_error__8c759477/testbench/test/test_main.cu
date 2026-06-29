#include "bilinear_interpolation.h"
#include <float.h>
#include <math.h>
#include <time.h>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

#undef NDEBUG
#include <assert.h>

void launch() {
    // Testcases count
    int testcases = 10;
    
    float threshold = 0.5f;

    // Input and output sizes
    int inputSizeArray[2][testcases] =  { {5, 3, 31, 18, 6,  30, 20, 26, 40, 28 }, { 5, 5, 14, 28, 10, 33, 16, 18, 38, 12 }};
    int outputSizeArray[2][testcases] = { {8, 7, 33, 44, 16, 32, 48, 43, 53, 33 }, { 8, 11, 37, 37, 29, 38, 29, 21, 50, 14 }};

    float tcase_1[25] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24 };
    float tcase_2[15] = { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12, 0.13, 0.14 };

    // Use a CUDA stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocating memory for the largest dataset
    int maxWidth = 100; int maxHeight = 100;
    float* inputMat_d; float* outputMat_d;
    CUDA_CHECK(cudaMallocAsync(&inputMat_d, maxWidth * maxHeight * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&outputMat_d, maxWidth * maxHeight * sizeof(float), stream));
	
	// Allocate memory for input and output
    float* outputMat_h = (float*)malloc(maxWidth * maxHeight * sizeof(float));
    float* inputMat_h= (float*)malloc(maxWidth * maxHeight * sizeof(float));
    float* outputMatExpected = (float*)malloc(maxWidth * maxHeight * sizeof(float));

    // Running testcases
    for (int tcase = 0; tcase < testcases; tcase++) {
        // Settings input and output dimensions
        int inputWidth = inputSizeArray[0][tcase];
        int inputHeight = inputSizeArray[1][tcase];
        int outputWidth = outputSizeArray[0][tcase];
        int outputHeight = outputSizeArray[1][tcase];

        if ((inputWidth > outputWidth) || (inputHeight > outputHeight)) {
            assert(false && "Output dimensions should be greater than input dimensions.");
        }

        // Generate random inputs in the range [-10,10]
        // Initializing random number state with deterministic seed.
        srand(42);
        for (int y = 0; y < inputHeight * inputWidth; y++) {
            if (tcase == 0) {
                inputMat_h[y] = tcase_1[y];
            } else if (tcase == 1) {
                inputMat_h[y] = tcase_2[y];
            } else {
                inputMat_h[y] = (float)(rand() % 100) / 100.f;
            }            
        }

        // Calling Bilinear interpolation on CPU
        float xRatio = (inputWidth - 1) / (float)(outputWidth - 1);
        float yRatio = (inputHeight - 1) / (float)(outputHeight - 1);
        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                float dx = x * xRatio;
                float dy = y * yRatio;

                int dx_l = floorf(dx); int dx_h = ceilf(dx);
                int dy_l = floorf(dy); int dy_h = ceilf(dy);

                float p00 = inputMat_h[dy_l * inputWidth + dx_l];
                float p01 = inputMat_h[dy_l * inputWidth + dx_h];
                float p10 = inputMat_h[dy_h * inputWidth + dx_l];
                float p11 = inputMat_h[dy_h * inputWidth + dx_h];

                float tx = dx - dx_l;
                float ty = dy - dy_l;
                float tmpX1 = ((1 - tx) * p00) + (tx * p01);
                float tmpX2 = ((1 - tx) * p10) + (tx * p11);
                float outVal = (1 - ty) * tmpX1 + ty * tmpX2;

                // Clip the outputs to the interval [0.0,1.0]
                outVal = (outVal > 1.0) ? 1.0 : (outVal < 0.0) ? 0.0 : outVal;
                outputMatExpected[y * outputWidth + x] = outVal;
            }
        }

        // Calling Bilinear interpolation on GPU
        // CUDA Initialization
        size_t shMemX = ceilf(BLOCK_SIZE * xRatio) + 2;
        size_t shMemY = ceilf(BLOCK_SIZE * yRatio) + 2;
        size_t totalShMemBytes = shMemX * shMemY * sizeof(float);

        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
        size_t blockSizeX = outputWidth / BLOCK_SIZE + 1;
        size_t blockSizeY = outputHeight / BLOCK_SIZE + 1;
        dim3 gridDim(blockSizeX, blockSizeY, 1);

        // Using pre-allocated memory to copy input to GPU memory
        CUDA_CHECK(cudaMemcpyAsync(inputMat_d, inputMat_h, inputWidth * inputHeight * sizeof(float), cudaMemcpyHostToDevice, stream));

        // CUDA kernel Launch
        void* args[] = { &inputMat_d, (void*)&inputWidth, (void*)&inputHeight, &outputMat_d, (void*)&outputWidth, (void*)&outputHeight };
        CUDA_CHECK(cudaLaunchKernel((void*)k_bilinearInterpolation, gridDim, blockDim, args, totalShMemBytes, stream));
        CUDA_CHECK(cudaMemcpyAsync(outputMat_h, outputMat_d, outputWidth * outputHeight * sizeof(float), cudaMemcpyDeviceToHost, stream));

        // Verification
        for (int y = 0; y < outputHeight; y++) {
            for (int x = 0; x < outputWidth; x++) {
                assert(fabsf(outputMat_h[y * outputWidth + x] - outputMatExpected[y * outputWidth + x]) < threshold);
            }
        }
    }
    
	// Free allocated memory
    free(inputMat_h);
    free(outputMatExpected);
    free(outputMat_h);

    CUDA_CHECK(cudaFreeAsync(inputMat_d, stream));
    CUDA_CHECK(cudaFreeAsync(outputMat_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    const int inputWidth = 2048;
    const int inputHeight = 2048;
    const int outputWidth = 4096;
    const int outputHeight = 4096;

    const int warmup = 3;
    const int timed = 100;

    size_t inBytes = (size_t)inputWidth * inputHeight * sizeof(float);
    size_t outBytes = (size_t)outputWidth * outputHeight * sizeof(float);

    float* inputMat_h = (float*)malloc(inBytes);
    srand(42);
    for (int i = 0; i < inputWidth * inputHeight; i++) {
        inputMat_h[i] = (float)(rand() % 100) / 100.f;
    }

    float* inputMat_d;
    float* outputMat_d;
    CUDA_CHECK(cudaMalloc(&inputMat_d, inBytes));
    CUDA_CHECK(cudaMalloc(&outputMat_d, outBytes));
    CUDA_CHECK(cudaMemcpy(inputMat_d, inputMat_h, inBytes, cudaMemcpyHostToDevice));

    float xRatio = (inputWidth - 1) / (float)(outputWidth - 1);
    float yRatio = (inputHeight - 1) / (float)(outputHeight - 1);

    size_t shMemX = ceilf(BLOCK_SIZE * xRatio) + 2;
    size_t shMemY = ceilf(BLOCK_SIZE * yRatio) + 2;
    size_t totalShMemBytes = shMemX * shMemY * sizeof(float);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridDim((outputWidth + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (outputHeight + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

    void* args[] = { &inputMat_d, (void*)&inputWidth, (void*)&inputHeight,
                     &outputMat_d, (void*)&outputWidth, (void*)&outputHeight };

    for (int i = 0; i < warmup; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_bilinearInterpolation,
                                    gridDim, blockDim, args, totalShMemBytes, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_bilinearInterpolation,
                                    gridDim, blockDim, args, totalShMemBytes, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    CUDA_CHECK(cudaFree(inputMat_d));
    CUDA_CHECK(cudaFree(outputMat_d));
    free(inputMat_h);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}