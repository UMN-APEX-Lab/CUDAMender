#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#undef NDEBUG
#include <assert.h>
#include "histogram.h"
#include "histogram_helpers.h"

void launch() {
    // Number of test cases
    constexpr int TEST_CASE_COUNT = 8;
    // Number of pixels in each test case
    int inputDataLength[TEST_CASE_COUNT] = {6, 8, 12, 12, 15, 8, 9, 16};
    //  Number of boundaries in each test case for each channel
    int numLevels[TEST_CASE_COUNT] = {5, 10, 8, 7, 12, 6, 5, 5};

    // Calculating maximum sizes
    const int MAX_NUM_PIXELS = *std::max_element(inputDataLength, inputDataLength + TEST_CASE_COUNT);
    const int maxNumLevels = *std::max_element(numLevels, numLevels + TEST_CASE_COUNT);

    if(MAX_NUM_LEVELS != maxNumLevels) {
      assert(false && "maxNumLevels is not equal to MAX_NUM_LEVELS");
    }

    // Input vectors
    uint8_t inputData_h[TEST_CASE_COUNT][MAX_NUM_PIXELS][NUM_CHANNELS] = {

        // 2 x 3 image size 
        {{2, 6, 7, 5}, {3, 0, 2, 1}, {7, 0, 6, 2}, 
        {0, 6, 7, 5}, {3, 0, 2, 6}, {167, 9, 217, 239}},

        // 2 x 4 image size
        {{108, 80, 41, 45}, {108, 24, 153, 120}, {178, 179, 163, 8}, {17, 81, 135, 167},
         {104, 209, 183, 247}, {136, 83, 27, 156}, {199, 108, 23, 68}, {167, 41, 30, 127}},

        // 3 x 4 image size
        {{39, 71, 112, 134}, {117, 224, 132, 241}, {163, 245, 61, 173}, {74, 171, 177, 17},
        {65, 57, 170, 216}, {88, 199, 172, 1}, {154, 99, 234, 0}, {118, 108, 117, 197}, 
        {82, 200, 120, 9}, {45, 184, 121, 39}, {102, 19, 61, 31}, {47, 61, 106, 12}},

        // 3 x 4 image size
        {{87, 155, 49, 189}, {62, 234, 68, 195}, {48, 73, 23, 147}, {174, 139, 108, 164},
         {165, 173, 162, 241}, {53, 181, 60, 30}, {155, 115, 117, 169}, {197, 89, 169, 106}, 
         {215, 213, 65, 157}, {149, 138, 222, 67}, {81, 30, 240, 165}, {122, 163, 139, 165}},

        // 3 x 5 image size
        {{139, 184, 133, 254}, {55, 27, 28, 16}, {103, 114, 93, 195}, {160, 197, 238, 249}, {49, 35, 178, 24},
         {134, 135, 220, 124}, {100, 171, 189, 133}, {89, 38, 150, 67}, {11, 193, 62, 113}, {176, 91, 188, 101},
         {174, 180, 113, 5}, {84, 108, 69, 50}, {210, 110, 227, 100}, {196, 101, 206, 193}, {96, 55, 202, 243}},

        // 2 x 4 image size
        { {83, 171, 112, 213}, {196, 42, 220, 253}, {131, 226, 150, 39}, {51, 104, 191, 211}, 
          {202, 81, 136, 23}, {28, 34, 173, 126}, {48, 126, 37, 14}, {217, 143, 237, 178}},

        // 3 x 3 image size
        { {149, 208, 225, 253}, {0, 221, 156, 253}, {135, 122, 205, 58}, 
          {127, 230, 147, 216}, {189, 150, 63, 170}, {21, 160, 169, 186}, 
          {228, 251, 196, 148}, {126, 199, 183, 231}, {228, 85, 178, 50}},

        // 4 x 4 image size
        { {237, 148, 4, 30}, {220, 123, 216, 53}, {141, 161, 8, 157}, {92, 12, 125, 49}, 
          {31, 52, 37, 48}, {10, 162, 72, 137}, {177, 127, 137, 113}, {31, 125, 218, 223}, 
          {69, 53, 144, 163}, {106, 52, 242, 21}, {44, 100, 212, 205}, {15, 102, 134, 106}, 
          {168, 160, 74, 110}, {3, 251, 42, 27}, {95, 50, 125, 86}, {243, 235, 13, 188}
         }
      };

    // Expected histogram outputs
    int expectedOutput[TEST_CASE_COUNT][NUM_ACTIVE_CHANNELS][MAX_NUM_BINS] = {
      { {5, 0, 1, 0}, {6, 0, 0, 0}, {5, 0, 0, 1} },
      { {1, 0, 0, 3, 1, 1, 2, 0, 0}, {1, 1, 3, 1, 0, 0, 1, 1, 0}, {2, 2, 0, 0, 1, 2, 1, 0, 0} },
      { {0, 4, 4, 2, 2, 0, 0}, {1, 3, 2, 0, 1, 3, 2}, {0, 2, 1, 5, 3, 0, 1} },
      { {0, 4, 2, 3, 2, 1}, {1, 1, 2, 4, 3, 1}, {1, 4, 2, 3, 0, 2} },
      { {1, 0, 2, 2, 3, 2, 1, 2, 1, 1, 0}, {0, 3, 1, 1, 4, 1, 0, 3, 2, 0, 0}, {0, 1, 2, 1, 1, 1, 1, 1, 4, 2, 1} },
      { {3, 1, 1, 2, 1}, {2, 1, 3, 1, 1}, {1, 0, 3, 2, 2} },
      { {2, 2, 3, 2}, {0, 2, 2, 5}, {1, 0, 5, 3} },
      { {6, 4, 3, 3}, {5, 5, 4, 2}, {5, 4, 3, 4} }
    };

    // Use a CUDA stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize host memories
    unsigned int lowerLevel_h[NUM_ACTIVE_CHANNELS] = {LOWER_LEVEL, LOWER_LEVEL, LOWER_LEVEL};
    unsigned int upperLevel_h[NUM_ACTIVE_CHANNELS] = {UPPER_LEVEL, UPPER_LEVEL, UPPER_LEVEL};
    int* histogram_h[NUM_ACTIVE_CHANNELS];
    // For each channel, allocate memory for the histogram
    for (int i = 0; i < NUM_ACTIVE_CHANNELS; ++i) {
        histogram_h[i] = (int*)malloc(MAX_NUM_BINS * sizeof(int));
    }

    // Allocating device memories for RGBA
    uint8_t* inputData_d;
    cudaMallocAsync(&inputData_d, MAX_NUM_PIXELS * NUM_CHANNELS * sizeof(uint8_t), stream);
    // Declare an array of device pointers to store histograms for each channel
    int* histogram_d[NUM_ACTIVE_CHANNELS];
    // Allocate memory for histograms for each channel R, G, B
    for (int i = 0; i < NUM_ACTIVE_CHANNELS; ++i) {
        cudaMallocAsync(&histogram_d[i], MAX_NUM_BINS * sizeof(int), stream);
    }

    // Loop to execute each test case
    for (int iTestcase = 0; iTestcase < TEST_CASE_COUNT; iTestcase++) {
        // Memory copying to device
        CUDA_CHECK(cudaMemcpyAsync(inputData_d, &inputData_h[iTestcase][0][0], sizeof(uint8_t) * inputDataLength[iTestcase] * NUM_CHANNELS, cudaMemcpyHostToDevice, stream));

        // Preparing numLevel for histogram
        int numLevelEachChannel = numLevels[iTestcase];
        int numLevelArray_h[NUM_ACTIVE_CHANNELS] = {numLevelEachChannel, numLevelEachChannel, numLevelEachChannel};

        calcMultiHistogram(inputData_d, histogram_d, numLevelArray_h, 
        lowerLevel_h, upperLevel_h, inputDataLength[iTestcase], stream);

        // Copy each channel's histogram data from device to host
        int numBins = numLevels[iTestcase] - 1;
        for (int i = 0; i < NUM_ACTIVE_CHANNELS; ++i) {
            cudaMemcpyAsync(histogram_h[i], histogram_d[i], numBins * sizeof(int), cudaMemcpyDeviceToHost, stream);
        }

        // Check tasks in the stream has completed
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Results verification
        for(int i = 0; i < numBins; i++) {
            assert(histogram_h[0][i] == expectedOutput[iTestcase][0][i]);
            assert(histogram_h[1][i] == expectedOutput[iTestcase][1][i]);
            assert(histogram_h[2][i] == expectedOutput[iTestcase][2][i]);
        }
    }

    for (int i = 0; i < NUM_ACTIVE_CHANNELS; ++i) {
        free(histogram_h[i]);
        CUDA_CHECK(cudaFreeAsync(histogram_d[i], stream));
    }
    CUDA_CHECK(cudaFreeAsync(inputData_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    constexpr int NUM_PIXELS = 4 * 1024 * 1024;
    constexpr int NUM_LEVELS = 257;
    constexpr int NUM_BINS = NUM_LEVELS - 1;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 100;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    uint8_t* inputData_h = (uint8_t*)malloc(NUM_PIXELS * NUM_CHANNELS * sizeof(uint8_t));
    for (int i = 0; i < NUM_PIXELS * NUM_CHANNELS; ++i) {
        inputData_h[i] = (uint8_t)(i % 256);
    }

    uint8_t* inputData_d;
    CUDA_CHECK(cudaMalloc(&inputData_d, NUM_PIXELS * NUM_CHANNELS * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(inputData_d, inputData_h, NUM_PIXELS * NUM_CHANNELS * sizeof(uint8_t), cudaMemcpyHostToDevice));

    int* histogram_d[NUM_ACTIVE_CHANNELS];
    for (int i = 0; i < NUM_ACTIVE_CHANNELS; ++i) {
        CUDA_CHECK(cudaMalloc(&histogram_d[i], NUM_BINS * sizeof(int)));
    }

    unsigned int lowerLevel_h[NUM_ACTIVE_CHANNELS] = {LOWER_LEVEL, LOWER_LEVEL, LOWER_LEVEL};
    unsigned int upperLevel_h[NUM_ACTIVE_CHANNELS] = {UPPER_LEVEL, UPPER_LEVEL, UPPER_LEVEL};
    int numLevels_h[NUM_ACTIVE_CHANNELS] = {NUM_LEVELS, NUM_LEVELS, NUM_LEVELS};

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        calcMultiHistogram(inputData_d, histogram_d, numLevels_h,
                           lowerLevel_h, upperLevel_h, NUM_PIXELS, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; ++i) {
        calcMultiHistogram(inputData_d, histogram_d, numLevels_h,
                           lowerLevel_h, upperLevel_h, NUM_PIXELS, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    for (int i = 0; i < NUM_ACTIVE_CHANNELS; ++i) {
        CUDA_CHECK(cudaFree(histogram_d[i]));
    }
    CUDA_CHECK(cudaFree(inputData_d));
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(inputData_h);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}