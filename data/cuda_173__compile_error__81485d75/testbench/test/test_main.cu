#include <cassert>
#include <cstdint>
#include <random>
#include <ctime>
#include <cmath>
#include <cstring>
#include <nvtx3/nvToolsExt.h>
#include "mma_tensor_conv.h"
#include "cuda_helpers.h"
#include "im2col_kernel.h"

#undef NDEBUG

// Test case struct to store the test case parameters
struct TestCase {
    int channels;
    int inputHeight;
    int inWidth;
    int kernelHeight;
    int kernelWidth;
    int pad;
    int stride;
    int numFilters;
};

// Host-side reference convolution.
// Computes N filters of an H×W image with C channels.
// — input : [C×H×W] row-major
// — weights: [N×C×KH×KW] row-major
// — output : [N×OH×OW] row-major
void refConv2dHost(const float* input,
                   const float* weights,
                   float* output,
                   int channels,
                   int inputHeight,
                   int inputWidth,
                   int numFilters,
                   int kernelHeight,
                   int kernelWidth,
                   int pad,
                   int stride,
                   int outHeight,
                   int outWidth) {
    int outSize = outHeight * outWidth;
    for (int f = 0; f < numFilters; ++f) {
        for (int oy = 0; oy < outHeight; ++oy) {
            for (int ox = 0; ox < outWidth; ++ox) {
                float sum = 0.0f;
                for (int c = 0; c < channels; ++c) {
                    for (int kh = 0; kh < kernelHeight; ++kh) {
                        for (int kw = 0; kw < kernelWidth; ++kw) {
                            int iy = oy*stride - pad + kh;
                            int ix = ox*stride - pad + kw;
                            float v = 0.0f;
                            if (iy >= 0 && iy < inputHeight && ix >= 0 && ix < inputWidth)
                                v = input[c*inputHeight*inputWidth + iy*inputWidth + ix];
                            float w = weights[
                                f*(channels*kernelHeight*kernelWidth)
                              + c*(kernelHeight*kernelWidth)
                              + kh*kernelWidth + kw
                            ];
                            sum += v * w;
                        }
                    }
                }
                output[f*outSize + oy*outWidth + ox] = sum;
            }
        }
    }
}

void launch() {
    const int TEST_CASE_COUNT = 9;
    const TestCase TEST_CASES[] = {
        {3, 16, 16, 3, 3, 1, 1, 16},     // RGB image, 3x3 kernel, 16 filters.
        {1, 8, 8, 3, 3, 1, 1, 16},       // Single channel 8x8 image, 3x3 kernel, 16 filters.
        {3, 32, 32, 5, 5, 2, 1, 16},     // RGB image, 5x5 kernel, pad=2, stride=1, 16 filters.
        {3, 64, 64, 3, 3, 1, 1, 32},     // Larger RGB image (64x64) with 3x3 kernel and 32 filters.
        {1, 128, 128, 5, 5, 2, 1, 64},   // Grayscale image (128x128) with 5x5 kernel, pad=2, stride=1, 64 filters.
        {3, 224, 224, 7, 7, 3, 2, 64},   // Standard RGB image (224x224) with 7x7 kernel, pad=3, stride=2, 64 filters.
        {3, 256, 256, 3, 3, 1, 1, 128},  // Large RGB image (256x256) with 3x3 kernel and 128 filters.
        {3, 40, 72, 3, 3, 1, 1, 32},     // Rectangular test: 40×72 RGB image, 3×3 kernel, pad=1, stride=1, 32 filters
        {1, 24, 48, 5, 5, 2, 1, 16}      // Rectangular Grayscale 24×48 image, 5×5 kernel, pad=2, stride=1, 16 filters
    };

    const float TOLERANCE = 0.01f;
    const int BLOCK_SIZE = 256;

    // Create a CUDA stream.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Setup random number generation with fixed seed for reproducibility.
    std::mt19937 randEngine(42); // Fixed seed
    std::uniform_real_distribution<float> randDist(0.0f, 1.0f);

    for (int tcIdx = 0; tcIdx < TEST_CASE_COUNT; tcIdx++) {
        TestCase testCase = TEST_CASES[tcIdx];
        int outHeight = (testCase.inputHeight + 2 * testCase.pad - testCase.kernelHeight) / testCase.stride + 1;
        int outWidth = (testCase.inWidth + 2 * testCase.pad - testCase.kernelWidth) / testCase.stride + 1;
        int im2colCols = outHeight * outWidth;
        int weightK = testCase.channels * testCase.kernelHeight * testCase.kernelWidth;

        // Choose padK as the next multiple of 16.
        int padK = ((weightK + 16 - 1) / 16) * 16;
        int convOutSize = testCase.numFilters * outHeight * outWidth;
        int inputSize = testCase.channels * testCase.inputHeight * testCase.inWidth;
        int weightSize = testCase.numFilters * weightK;

        // Host arrays.
        float *input_h = new float[inputSize];
        float *weights_h = new float[weightSize];
        float *convStd_h = new float[convOutSize];

        // Random input (image) data
        for (int j = 0; j < inputSize; j++) {
            float val = randDist(randEngine);
            input_h[j] = __nv_bfloat16(val);
        }

        // Random weight kernel
        for (int j = 0; j < weightSize; j++) {
            float val = randDist(randEngine);
            weights_h[j] = __nv_bfloat16(val);
        }

        // Device arrays (using asynchronous allocations).
        float *input_d;
        float *weights_d;
        CUDA_CHECK(cudaMallocAsync(&input_d, inputSize * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&weights_d, weightSize * sizeof(float), stream));

        refConv2dHost(
            input_h, weights_h, convStd_h,
            testCase.channels,
            testCase.inputHeight, testCase.inWidth,
            testCase.numFilters,
            testCase.kernelHeight, testCase.kernelWidth,
            testCase.pad, testCase.stride,
            outHeight, outWidth
        );

        // Convert input and weights to BF16.
        __nv_bfloat16 *inputBf16_h = new __nv_bfloat16[inputSize];
        __nv_bfloat16 *weightsBf16_h = new __nv_bfloat16[weightSize];
        for (int j = 0; j < inputSize; j++) {
            inputBf16_h[j] = __nv_bfloat16(input_h[j]);
        }
        for (int j = 0; j < weightSize; j++) {
            weightsBf16_h[j] = __nv_bfloat16(weights_h[j]);
        }

        __nv_bfloat16 *inputBf16_d;
        __nv_bfloat16 *weightsBf16_d;
        CUDA_CHECK(cudaMallocAsync(&inputBf16_d, inputSize * sizeof(__nv_bfloat16), stream));
        CUDA_CHECK(cudaMallocAsync(&weightsBf16_d, weightSize * sizeof(__nv_bfloat16), stream));
        CUDA_CHECK(cudaMemcpyAsync(inputBf16_d, inputBf16_h, inputSize * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(weightsBf16_d, weightsBf16_h, weightSize * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));

        // Allocate output arrays for preprocessing.
        int im2colPadSize = padK * im2colCols;       // For im2col.
        int weightPadSize = testCase.numFilters * padK;  // For weights.
        __nv_bfloat16 *im2colPad_d;
        __nv_bfloat16 *weightPad_d;

        CUDA_CHECK(cudaMallocAsync(&im2colPad_d, im2colPadSize * sizeof(__nv_bfloat16), stream));
        CUDA_CHECK(cudaMallocAsync(&weightPad_d, weightPadSize * sizeof(__nv_bfloat16), stream));

        // Launch the im2col preprocessing kernel.
        int totalPreprocess = weightPadSize + im2colPadSize;
        int gridSize = (totalPreprocess + BLOCK_SIZE - 1) / BLOCK_SIZE;
        k_im2ColTransform<<<gridSize, BLOCK_SIZE, 0, stream>>>(inputBf16_d, weightsBf16_d,
                                                               im2colPad_d, weightPad_d,
                                                               testCase.channels, testCase.inputHeight, testCase.inWidth,
                                                               testCase.kernelHeight, testCase.kernelWidth,
                                                               testCase.pad, testCase.stride,
                                                               outHeight, outWidth,
                                                               testCase.numFilters, padK);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Allocate output buffer for the MMA kernel.
        float *convMma_d;
        CUDA_CHECK(cudaMallocAsync(&convMma_d, convOutSize * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(convMma_d, 0, convOutSize * sizeof(float), stream));

        //The MMA kernel expects dimensions:
        //mDim = numFilters, nDim = im2colCols, kDim = padK.
        int M = testCase.numFilters;
        int N = im2colCols;
        int K = padK;

        assert(M % MMA_M == 0);
        assert(N % MMA_N == 0);
        assert(K % MMA_K == 0);

        // Kernel Launch
        // blockDim -> (256,1,1)
        // gridDim -> ((N + MMA_N -1)/MMA_N, (M + MMA_M - 1)/MMA_M))
        dim3 gridDimMma((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M);
        dim3 blockDimMma(256);
        int shmemBytes = (MMA_M * MMA_K + MMA_K * MMA_N) * sizeof(__nv_bfloat16);

        // Launch the MMA GEMM kernel: weightPad_d (mDim×kDim) multiplied by im2colPad_d (kDim×nDim)
        void *kernelArgs[] = { &weightPad_d, &im2colPad_d, &convMma_d,
                               (void*)&M, &N, &K };
        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaTensorConvMatMul,
                                    gridDimMma,
                                    blockDimMma,
                                    kernelArgs,
                                    shmemBytes,
                                    stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        //Copy MMA result and reference convolution result back to host.
        float *convMma_h  = new float[convOutSize];
        CUDA_CHECK(cudaMemcpyAsync(convMma_h , convMma_d, convOutSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        //Verify results.
        for (int f = 0; f < testCase.numFilters; f++) {
            for (int j = 0; j < outHeight * outWidth; j++) {
                float stdVal = convStd_h[f * (outHeight * outWidth) + j];
                float mmaVal = convMma_h[f * (outHeight * outWidth) + j];
                float diff = fabs(stdVal - mmaVal);
                assert(fabs(stdVal) > 1e-5f && diff / fabs(stdVal) < TOLERANCE);
            }
        }

        //Memory Clean up
        CUDA_CHECK(cudaFreeAsync(input_d, stream));
        CUDA_CHECK(cudaFreeAsync(weights_d, stream));
        CUDA_CHECK(cudaFreeAsync(inputBf16_d, stream));
        CUDA_CHECK(cudaFreeAsync(weightsBf16_d, stream));
        CUDA_CHECK(cudaFreeAsync(im2colPad_d, stream));
        CUDA_CHECK(cudaFreeAsync(weightPad_d, stream));
        CUDA_CHECK(cudaFreeAsync(convMma_d, stream));

        // Free host memory.
        delete[] input_h;
        delete[] weights_h;
        delete[] inputBf16_h;
        delete[] weightsBf16_h;
        delete[] convStd_h;
        delete[] convMma_h ;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    const int BLOCK_SIZE = 256;

    // Large workload: 3-channel 1024x1024 image, 3x3 kernel, pad=1, stride=1, 256 filters
    const int channels = 3, inputHeight = 1024, inputWidth = 1024;
    const int kernelHeight = 3, kernelWidth = 3;
    const int pad = 1, stride = 1, numFilters = 256;
    const int WARMUP = 3;
    const int TIMED = 200;

    int outHeight = (inputHeight + 2 * pad - kernelHeight) / stride + 1;
    int outWidth  = (inputWidth  + 2 * pad - kernelWidth)  / stride + 1;
    int im2colCols = outHeight * outWidth;
    int weightK    = channels * kernelHeight * kernelWidth;
    int padK       = ((weightK + 16 - 1) / 16) * 16;
    int convOutSize = numFilters * outHeight * outWidth;
    int inputSize  = channels * inputHeight * inputWidth;
    int weightSize = numFilters * weightK;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::mt19937 randEngine(123);
    std::uniform_real_distribution<float> randDist(0.0f, 1.0f);

    __nv_bfloat16 *inputBf16_h = new __nv_bfloat16[inputSize];
    __nv_bfloat16 *weightsBf16_h = new __nv_bfloat16[weightSize];
    for (int j = 0; j < inputSize; j++)
        inputBf16_h[j] = __nv_bfloat16(randDist(randEngine));
    for (int j = 0; j < weightSize; j++)
        weightsBf16_h[j] = __nv_bfloat16(randDist(randEngine));

    __nv_bfloat16 *inputBf16_d, *weightsBf16_d;
    CUDA_CHECK(cudaMallocAsync(&inputBf16_d, inputSize * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&weightsBf16_d, weightSize * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMemcpyAsync(inputBf16_d, inputBf16_h, inputSize * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(weightsBf16_d, weightsBf16_h, weightSize * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice, stream));

    int im2colPadSize = padK * im2colCols;
    int weightPadSize = numFilters * padK;
    __nv_bfloat16 *im2colPad_d, *weightPad_d;
    CUDA_CHECK(cudaMallocAsync(&im2colPad_d, im2colPadSize * sizeof(__nv_bfloat16), stream));
    CUDA_CHECK(cudaMallocAsync(&weightPad_d, weightPadSize * sizeof(__nv_bfloat16), stream));

    int totalPreprocess = weightPadSize + im2colPadSize;
    int gridSize = (totalPreprocess + BLOCK_SIZE - 1) / BLOCK_SIZE;
    k_im2ColTransform<<<gridSize, BLOCK_SIZE, 0, stream>>>(
        inputBf16_d, weightsBf16_d, im2colPad_d, weightPad_d,
        channels, inputHeight, inputWidth,
        kernelHeight, kernelWidth, pad, stride,
        outHeight, outWidth, numFilters, padK);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float *convMma_d;
    CUDA_CHECK(cudaMallocAsync(&convMma_d, convOutSize * sizeof(float), stream));

    int M = numFilters;
    int N = im2colCols;
    int K = padK;
    dim3 gridDimMma((N + MMA_N - 1) / MMA_N, (M + MMA_M - 1) / MMA_M);
    dim3 blockDimMma(256);
    int shmemBytes = (MMA_M * MMA_K + MMA_K * MMA_N) * sizeof(__nv_bfloat16);

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        CUDA_CHECK(cudaMemsetAsync(convMma_d, 0, convOutSize * sizeof(float), stream));
        void *kernelArgs[] = { &weightPad_d, &im2colPad_d, &convMma_d,
                               (void*)&M, &N, &K };
        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaTensorConvMatMul,
                                    gridDimMma, blockDimMma,
                                    kernelArgs, shmemBytes, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed region
    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED; i++) {
        CUDA_CHECK(cudaMemsetAsync(convMma_d, 0, convOutSize * sizeof(float), stream));
        void *kernelArgs[] = { &weightPad_d, &im2colPad_d, &convMma_d,
                               (void*)&M, &N, &K };
        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaTensorConvMatMul,
                                    gridDimMma, blockDimMma,
                                    kernelArgs, shmemBytes, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(inputBf16_d, stream));
    CUDA_CHECK(cudaFreeAsync(weightsBf16_d, stream));
    CUDA_CHECK(cudaFreeAsync(im2colPad_d, stream));
    CUDA_CHECK(cudaFreeAsync(weightPad_d, stream));
    CUDA_CHECK(cudaFreeAsync(convMma_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    delete[] inputBf16_h;
    delete[] weightsBf16_h;
}

int main(int argc, char* argv[]) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
    return 0;
}