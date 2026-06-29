#include "graph_pipeline.h"
#include <cstring>
#include <nvtx3/nvToolsExt.h>

// Verification function CPU sequential implementation
void verificationFunction(float* inpFrame, int frameHeight, int frameWidth, size_t numFrames, float gammaValue, float contrastValue, float brightnessValue, float* outFrame) {

    size_t numElementsInFrame = frameHeight * frameWidth;
    float* inpFramePointer = inpFrame;
    float* outFramePointer = outFrame;
    int inputFrameOffset = numElementsInFrame * RGB_CHANNELS;
    int outputFrameOffset = numElementsInFrame;

    // Running operations per frame
    for (int frameIdx = 0;frameIdx < numFrames;frameIdx++) {
        inpFramePointer = inpFrame + frameIdx * inputFrameOffset;
        outFramePointer = outFrame + frameIdx * outputFrameOffset;

        // RGB 2 Grayscale and gamma correction
        for (int r = 0;r < frameHeight;r++) {
            for (int c = 0;c < frameWidth;c++) {

                // RGB to Grayscale
                float pixValue = inpFramePointer[r * frameWidth + c] * RGB2GRAY_R +
                    inpFramePointer[r * frameWidth + c + numElementsInFrame] * RGB2GRAY_G +
                    inpFramePointer[r * frameWidth + c + 2 * numElementsInFrame] * RGB2GRAY_B;
                pixValue = std::min(std::max(pixValue, 0.f), 1.f);

                // Gamma, contrast and brightness
                pixValue = std::max(0.f, std::min(1.f, pow(pixValue, gammaValue)));
                pixValue = contrastValue * pixValue + brightnessValue;
                outFramePointer[r * frameWidth + c] = std::min(std::max(pixValue, 0.f), 1.f);
            }
        }
    }
}

void launch() {

    // Setting input lengths
    const int NUM_TESTCASES = 8;
    const float VERIFICATION_TOLERANCE = 1e-3;
    
    // Testcases select input height, width and frame count randomly
    const int MAX_WIDTH = 100;
    const int MAX_HEIGHT = 100;
    const int MAX_FRAME_COUNT = 7;
    int testcases[NUM_TESTCASES][3] = { {3,3,2}, {5,5,2}, {25,24,5}, {27,36,4}, {78,100,3}, {81,27,2}, {64,57,7}, {50,66,3} };

    // Memory allocation in CPU
    size_t maxElementsInFrame = MAX_WIDTH * MAX_HEIGHT * MAX_FRAME_COUNT;
    float* inpFrame = (float*)malloc(RGB_CHANNELS * maxElementsInFrame * sizeof(float));
    float* outFrame_h = (float*)malloc(maxElementsInFrame * sizeof(float));
    float* outFrame_ref = (float*)malloc(maxElementsInFrame * sizeof(float));

    // Initializing stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Allocating device memory and copying host matrices to device
    float* inpFrame_d = nullptr;
    float* outFrame_d = nullptr;
    float* outRGB2Gray_d = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&inpFrame_d, RGB_CHANNELS * maxElementsInFrame * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&outRGB2Gray_d, maxElementsInFrame * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&outFrame_d, maxElementsInFrame * sizeof(float), stream));

    // Running test cases with random inputs
    srand(42);
    for (int tIter = 0; tIter < NUM_TESTCASES; tIter++) {

        int frameWidth = testcases[tIter][0];
        int frameHeight = testcases[tIter][1];
        size_t numFrames = testcases[tIter][2];
        size_t numElementsInFrame = frameHeight * frameWidth;

        // Set input gamma, contrast and brightness
        float gammaValue = (rand() % 1000) / 1000.f * 2.f;
        float contrastValue = (rand() % 1000) / 1000.f;
        float brightnessValue = (rand() % 1000) / 1000.f;

        // Input data generation for frame in the interval (-1,1)
        for (int iter = 0; iter < RGB_CHANNELS * numElementsInFrame * numFrames; iter++)
            inpFrame[iter] = (rand() % 1000) / 1000.f;

        // Generate CUDA graphs
        runGraph(inpFrame, inpFrame_d, frameHeight, frameWidth, numFrames, gammaValue, contrastValue, brightnessValue, outRGB2Gray_d, outFrame_d, outFrame_h, stream);

        // Verify the output with CPU implementation
        verificationFunction(inpFrame, frameHeight, frameWidth, numFrames, gammaValue, contrastValue, brightnessValue, outFrame_ref);
        for (int i = 0; i < numElementsInFrame * numFrames; i++) {
            assert(fabsf(outFrame_h[i] - outFrame_ref[i]) < VERIFICATION_TOLERANCE);
        }
    }

    // Free allocated memory
    free(inpFrame);
    free(outFrame_h);
    free(outFrame_ref);

    CUDA_CHECK(cudaFreeAsync(inpFrame_d, stream));
    CUDA_CHECK(cudaFreeAsync(outRGB2Gray_d, stream));
    CUDA_CHECK(cudaFreeAsync(outFrame_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    const int WARMUP_ITERS = 3;
    const int TIMED_ITERS = 100;

    const int frameWidth = 1024;
    const int frameHeight = 1024;
    const size_t numFrames = 20;
    const size_t numElementsInFrame = frameHeight * frameWidth;

    float gammaValue = 0.472f;
    float contrastValue = 0.104f;
    float brightnessValue = 0.724f;

    float* inpFrame = (float*)malloc(RGB_CHANNELS * numElementsInFrame * numFrames * sizeof(float));
    float* outFrame_h = (float*)malloc(numElementsInFrame * numFrames * sizeof(float));

    srand(42);
    for (size_t i = 0; i < RGB_CHANNELS * numElementsInFrame * numFrames; i++)
        inpFrame[i] = (rand() % 1000) / 1000.f;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float* inpFrame_d = nullptr;
    float* outFrame_d = nullptr;
    float* outRGB2Gray_d = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&inpFrame_d, RGB_CHANNELS * numElementsInFrame * numFrames * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&outRGB2Gray_d, numElementsInFrame * numFrames * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&outFrame_d, numElementsInFrame * numFrames * sizeof(float), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < WARMUP_ITERS; i++) {
        runGraph(inpFrame, inpFrame_d, frameHeight, frameWidth, numFrames,
                 gammaValue, contrastValue, brightnessValue,
                 outRGB2Gray_d, outFrame_d, outFrame_h, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        runGraph(inpFrame, inpFrame_d, frameHeight, frameWidth, numFrames,
                 gammaValue, contrastValue, brightnessValue,
                 outRGB2Gray_d, outFrame_d, outFrame_h, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    free(inpFrame);
    free(outFrame_h);

    CUDA_CHECK(cudaFreeAsync(inpFrame_d, stream));
    CUDA_CHECK(cudaFreeAsync(outRGB2Gray_d, stream));
    CUDA_CHECK(cudaFreeAsync(outFrame_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}