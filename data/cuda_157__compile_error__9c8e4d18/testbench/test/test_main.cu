#include "cdf_calculator.h"
#include "cuda_utils.h"
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cstring>
#undef NDEBUG
#include <cassert>
#include <nvtx3/nvToolsExt.h>

#define EPSILON 1e-4

void launch() {
    // Number of test cases
    constexpr int TEST_CASE_COUNT = 7;
    int inputDataLength[TEST_CASE_COUNT] = {16, 16, 16, 16, 35, 42, 64};
    const int MAX_VECTOR_SIZE = *std::max_element(inputDataLength, inputDataLength + TEST_CASE_COUNT);
    // input image in row major layout
    int srcData_h[TEST_CASE_COUNT][MAX_VECTOR_SIZE] = {
        // 4x4 image
        {   61, 112, 91, 176,
            134, 154, 32, 68,
            84, 163, 31, 58,
            135, 162, 163, 52},
        // 4x4 image  
        {   18, 171, 45, 34,
            239, 61, 73, 52,
            82, 57, 245, 228,
            102, 31, 205, 193},
        // 4x4 image
        {   211, 246, 150, 18,
            117, 42, 152, 25,
            199, 108, 163, 129,
            254, 165, 234, 115},
        // 4x4 image
        {   208, 208, 251, 85,
            229, 113, 75, 58,
            253, 84, 149, 225,
            69, 40, 187, 175},
        // 7x5 image
        {   125, 125, 86, 230, 94,
            28, 199, 99, 61, 103,
            24, 33, 241, 244, 147,
            15, 60, 90, 210, 3,
            11, 43, 166, 187, 165,
            115, 140, 75, 190, 48,
            175, 46, 94, 160, 199},
        // 6x7 image
        {   85, 174, 34, 184, 27, 167, 126,
            199, 183, 231, 228, 85, 178, 50,
            7, 190, 128, 122, 231, 156, 158,
            220, 206, 147, 46, 61, 226, 7,
            125, 42, 250, 182, 128, 120, 15,
            174, 10, 18, 133, 24, 209, 209},
        // 4x16 image
        {192, 112, 211, 140, 193, 254, 185, 158, 222,  41, 218,  17,  54,  21, 230,  98,
         174,  31, 248, 162, 149, 109,  23, 115, 215, 205, 219,  28, 121, 148,  70, 229,
          50, 162,  51,  83, 238,  54,  24,  56, 93, 216, 124,  11, 247, 137,  80,  61,
          25,  71,  75,   1, 203, 236, 128,  75, 123, 133,  62, 198, 144,  87, 140, 183
         }
    };

    // expected output in row major layout
    float expectedOutput[TEST_CASE_COUNT][MAX_VECTOR_SIZE] = {
        // 4x4 image
        {   0.3125, 0.5625, 0.5   , 1.   ,
            0.625 , 0.75  , 0.125 , 0.375 ,
            0.4375, 0.9375, 0.0625, 0.25  ,
            0.6875, 0.8125, 0.9375, 0.1875},
        // 4x4 image
        {   0.0625, 0.6875, 0.25  , 0.1875,
            0.9375, 0.4375, 0.5   , 0.3125,
            0.5625, 0.375 , 1.    , 0.875 ,
            0.625 , 0.125 , 0.8125, 0.75  },
        // 4x4 image
        {   0.8125, 0.9375, 0.5   , 0.0625,
            0.375 , 0.1875, 0.5625, 0.125 ,
            0.75  , 0.25  , 0.625 , 0.4375,
            1.    , 0.6875, 0.875 , 0.3125},
        // 4x4 image
        {   0.75  , 0.75  , 0.9375, 0.375,
            0.875 , 0.4375, 0.25  , 0.125 ,
            1.    , 0.3125, 0.5   , 0.8125,
            0.1875, 0.0625, 0.625 , 0.5625},
        // 7x5 image
        {   0.6000, 0.6000, 0.3714, 0.9429, 0.4571,
            0.1429, 0.8857, 0.4857, 0.3143, 0.5143,
            0.1143, 0.1714, 0.9714, 1.0000, 0.6571,
            0.0857, 0.2857, 0.4000, 0.9143, 0.0286,
            0.0571, 0.2000, 0.7429, 0.8000, 0.7143,
            0.5429, 0.6286, 0.3429, 0.8286, 0.2571,
            0.7714, 0.2286, 0.4571, 0.6857, 0.8857},
        // 6x7 image
        {   0.3333, 0.6429, 0.1905, 0.7381, 0.1667, 0.5952, 0.4286,
            0.7857, 0.7143, 0.9762, 0.9286, 0.3333, 0.6667, 0.2619,
            0.0476, 0.7619, 0.4762, 0.3810, 0.9762, 0.5476, 0.5714,
            0.8810, 0.8095, 0.5238, 0.2381, 0.2857, 0.9048, 0.0476,
            0.4048, 0.2143, 1.0000, 0.6905, 0.4762, 0.3571, 0.0952,
            0.6429, 0.0714, 0.1190, 0.5000, 0.1429, 0.8571, 0.8571},
        // 4x16 image
        {   0.734375, 0.4375  , 0.8125  , 0.578125, 0.75    , 1.      , 0.71875 , 0.640625, 0.890625, 0.15625 , 0.859375, 0.046875, 0.21875 , 0.0625  , 0.921875, 0.40625 ,
            0.6875  , 0.140625, 0.984375, 0.671875, 0.625   , 0.421875, 0.078125, 0.453125, 0.828125, 0.796875, 0.875   , 0.125   , 0.46875 , 0.609375, 0.28125 , 0.90625 ,
            0.171875, 0.671875, 0.1875  , 0.359375, 0.953125, 0.21875 , 0.09375 , 0.234375, 0.390625, 0.84375 , 0.5     , 0.03125 , 0.96875 , 0.546875, 0.34375 , 0.25    ,
            0.109375, 0.296875, 0.328125, 0.015625, 0.78125 , 0.9375  , 0.515625, 0.328125, 0.484375, 0.53125 , 0.265625, 0.765625, 0.59375 , 0.375   , 0.578125, 0.703125}
    };

    // Use a CUDA stream for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Initialize result on the host
    std::vector<float> cdf_h(MAX_VECTOR_SIZE);

    // Vectors for device memory
    thrust::device_vector<float> cdf_d(MAX_VECTOR_SIZE);
    thrust::device_vector<int> srcImage_d(MAX_VECTOR_SIZE);

    // Loop to execute each test case
    for (int iTestcase = 0; iTestcase < TEST_CASE_COUNT; iTestcase++) {
        
        // Copy host data into device memory
        CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(srcImage_d.data()), srcData_h[iTestcase], inputDataLength[iTestcase] * sizeof(int), cudaMemcpyHostToDevice, stream));

        // Execute kernel
        calculatePixelCdf(srcImage_d, inputDataLength[iTestcase], stream, cdf_d);

        // Copy device data into host memory
        CUDA_CHECK(cudaMemcpyAsync(cdf_h.data(), thrust::raw_pointer_cast(cdf_d.data()), inputDataLength[iTestcase] * sizeof(float), cudaMemcpyDeviceToHost, stream));

        // Synchronize the stream to ensure all operations are complete
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Verifying result with expected output
        for (unsigned i=0; i < inputDataLength[iTestcase]; i++) {
            assert(abs(cdf_h[i] - expectedOutput[iTestcase][i]) < EPSILON);
        }
    }
}

void benchmark() {
    const int SIZE = 2 * 1024 * 1024;
    const int WARMUP = 3;
    const int ITERS = 100;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<int> src_h(SIZE);
    srand(42);
    for (int i = 0; i < SIZE; i++) {
        src_h[i] = rand() % 256;
    }

    thrust::device_vector<int> srcImage_d(SIZE);
    thrust::device_vector<float> cdf_d(SIZE);
    std::vector<int> src_copy(SIZE);

    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(srcImage_d.data()),
                               src_h.data(), SIZE * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int i = 0; i < WARMUP; i++) {
        CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(srcImage_d.data()),
                                   src_h.data(), SIZE * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        calculatePixelCdf(srcImage_d, SIZE, stream, cdf_d);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < ITERS; i++) {
        CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(srcImage_d.data()),
                                   src_h.data(), SIZE * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        calculatePixelCdf(srcImage_d, SIZE, stream, cdf_d);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    cudaStreamDestroy(stream);
}

int main(int argc, char* argv[]) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}