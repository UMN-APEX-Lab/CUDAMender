#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <cstring>
#include <random>
#include <nvtx3/nvToolsExt.h>
#include "blur_kernel.h"

// Using initializer lists to initialize cuda::std::array.
cuda::std::array<float, BLUR_CALCULATIONS_PER_PIXEL> blurCoefficients_h = { COEFFICIENT_0, COEFFICIENT_1, COEFFICIENT_2, COEFFICIENT_3, COEFFICIENT_4, COEFFICIENT_5, COEFFICIENT_6, COEFFICIENT_7, COEFFICIENT_8 };
// Blur indices.
cuda::std::array<int, BLUR_CALCULATIONS_PER_PIXEL> blurOffsetX_h = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
cuda::std::array<int, BLUR_CALCULATIONS_PER_PIXEL> blurOffsetY_h = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

void launch() {
    constexpr float ERROR_TOLERANCE = 1e-2f;
    constexpr int DETERMINISTIC_RANDOM_NUMBER_SEED = 42;
    constexpr float WAVE_FREQUENCY = 0.01f;
    
    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));
    size_t dynamicSharedMemorySize = 0;
    int minGridSize;
    int blockSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)k_blur, dynamicSharedMemorySize));
    
    // Allocating host memory.
    cuda::std::array<float, MAX_PIXELS>* pixel_h = new cuda::std::array<float, MAX_PIXELS>();
    cuda::std::array<float, MAX_PIXELS>* blurredPixel_h = new cuda::std::array<float, MAX_PIXELS>();

    constexpr int NUM_EXAMPLES = 5;
    int* testWidth = new int[NUM_EXAMPLES];
    int* testHeight = new int[NUM_EXAMPLES];
    float* testInput = new float[MAX_PIXELS * NUM_EXAMPLES];

    // Allocating device memory.
    cuda::std::array<float, MAX_PIXELS>* pixel_d;
    cuda::std::array<float, MAX_PIXELS>* blurredPixel_d;
    CUDA_CHECK(cudaMallocAsync(&pixel_d, sizeof(cuda::std::array<float, MAX_PIXELS>), stream));
    CUDA_CHECK(cudaMallocAsync(&blurredPixel_d, sizeof(cuda::std::array<float, MAX_PIXELS>), stream));
    
    // Test 1
    {
        constexpr int WIDTH = 5;
        constexpr int HEIGHT = 4;
        int width = WIDTH;
        int height = HEIGHT;
        assert(WIDTH * HEIGHT <= MAX_PIXELS);
        cuda::std::array<float, WIDTH * HEIGHT> input = { 0.00f, 1.00f, 0.00f, 1.00f, 0.00f, 1.00f, 0.00f, 1.00f, 0.00f, 1.00f, 0.00f, 1.00f, 0.00f, 1.00f, 0.00f, 1.00f, 0.00f, 1.00f, 0.00f, 1.00f };
        cuda::std::array<float, WIDTH * HEIGHT> expectedOutput = { 0.250f, 0.375f, 0.375f, 0.375f, 0.250f, 0.375f, 0.500f, 0.500f, 0.500f, 0.375f, 0.375f, 0.500f, 0.500f, 0.500f, 0.375f, 0.312f, 0.375f, 0.375f, 0.375f, 0.312f };
        for(int i = 0; i < width * height; i++) {
            (*pixel_h)[i] = input[i];
        }
        // Blurring the pixels.
        CUDA_CHECK(cudaMemcpyAsync(pixel_d, pixel_h, sizeof(float) * width * height, cudaMemcpyHostToDevice, stream));
        void* args[4] = { &width, &height, &pixel_d, &blurredPixel_d };
        int requiredBlocks = (width * height + blockSize - 1) / blockSize;
        int usedBlocks = requiredBlocks < minGridSize ? requiredBlocks : minGridSize;
        // Grid: (usedBlocks, 1, 1)
        // Block: (blockSize, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_blur, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, dynamicSharedMemorySize, stream));
        CUDA_CHECK(cudaMemcpyAsync(blurredPixel_h, blurredPixel_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < width * height; i++) {
            assert(fabsf(expectedOutput[i] - (*blurredPixel_h)[i]) < ERROR_TOLERANCE);
        }
    }
    // Test 2
    {
        constexpr int WIDTH = 4;
        constexpr int HEIGHT = 5;
        int width = WIDTH;
        int height = HEIGHT;
        assert(width * height <= MAX_PIXELS);
        cuda::std::array<float, WIDTH * HEIGHT> input = { 0.00f, 0.05f, 0.10f, 0.15f, 0.20f, 0.25f, 0.30f, 0.35f, 0.40f, 0.45f, 0.50f, 0.55f, 0.60f, 0.65f, 0.70f, 0.75f, 0.80f, 0.85f, 0.90f, 0.95f };
        cuda::std::array<float, WIDTH * HEIGHT> expectedOutput = { 0.047f, 0.088f, 0.125f, 0.112f, 0.163f, 0.250f, 0.300f, 0.250f, 0.313f, 0.450f, 0.500f, 0.400f, 0.462f, 0.650f, 0.700f, 0.550f, 0.422f, 0.587f, 0.625f, 0.488f };
        for(int i = 0; i < width * height; i++) {
            (*pixel_h)[i] = input[i];
        }
        // Blurring the pixels.
        CUDA_CHECK(cudaMemcpyAsync(pixel_d, pixel_h, sizeof(float) * width * height, cudaMemcpyHostToDevice, stream));
        void* args[4] = { &width, &height, &pixel_d, &blurredPixel_d };
        int requiredBlocks = (width * height + blockSize - 1) / blockSize;
        int usedBlocks = requiredBlocks < minGridSize ? requiredBlocks : minGridSize;
        // Grid: (usedBlocks, 1, 1)
        // Block: (blockSize, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_blur, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, dynamicSharedMemorySize, stream));
        CUDA_CHECK(cudaMemcpyAsync(blurredPixel_h, blurredPixel_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < width * height; i++) {
            assert(fabsf(expectedOutput[i] - (*blurredPixel_h)[i]) < ERROR_TOLERANCE);
        }
    }
    
    // Preparing data for tests 3, 4, 5, 6, and 7
    int testIndex = 0;
    // Test 3 initialization.
    {
        int width = 100;
        int height = 100;
        testWidth[testIndex] = width;
        testHeight[testIndex] = height;
        assert(width * height <= MAX_PIXELS);
        std::mt19937 generator(DETERMINISTIC_RANDOM_NUMBER_SEED);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        for(int i = 0; i < width * height; i++) {
            testInput[i + testIndex * MAX_PIXELS] = distribution(generator);
        }
        testIndex++;
    }
    // Test 4 initialization.
    {
        int width = 80;
        int height = 60;
        testWidth[testIndex] = width;
        testHeight[testIndex] = height;
        assert(width * height <= MAX_PIXELS);
        for(int i = 0; i < width * height; i++) {
            testInput[i + testIndex * MAX_PIXELS] = cos((i % width) * WAVE_FREQUENCY) * sin((i / width) * WAVE_FREQUENCY);
        }
        testIndex++;
    }
    // Test 5 initialization.
    {
        int width = 64;
        int height = 48;
        testWidth[testIndex] = width;
        testHeight[testIndex] = height;
        assert(width * height <= MAX_PIXELS);
        for(int i = 0; i < width * height; i++) {
            float dx = (i % width) - (width / 2);
            float dy = (i / width) - (height / 2);
            float r = sqrt(dx * dx + dy * dy) + 1.0f;
            testInput[i + testIndex * MAX_PIXELS] = 1.0f / r;
        }
        testIndex++;
    }
    // Test 6 initialization.
    {
        int width = 40;
        int height = 30;
        testWidth[testIndex] = width;
        testHeight[testIndex] = height;
        assert(width * height <= MAX_PIXELS);
        for(int i = 0; i < width * height; i++) {
            testInput[i + testIndex * MAX_PIXELS] = 1.0f;
        }
        testIndex++;
    }
    // Test 7 initialization.
    {
        int width = 34;
        int height = 54;
        testWidth[testIndex] = width;
        testHeight[testIndex] = height;
        assert(width * height <= MAX_PIXELS);
        for(int i = 0; i < width * height; i++) {
            testInput[i + testIndex * MAX_PIXELS] = ((i % 3) == 0);
        }
        testIndex++;
    }
    // Iterating tests 3, 4, 5, 6, and 7.
    for(int test = 0; test < testIndex; test++)
    {
        int width = testWidth[test];
        int height = testHeight[test];
        for(int i = 0; i < width * height; i++) {
            (*pixel_h)[i] = testInput[i + test * MAX_PIXELS];
        }
        // Blurring the pixels.
        CUDA_CHECK(cudaMemcpyAsync(pixel_d, pixel_h, sizeof(float) * width * height, cudaMemcpyHostToDevice, stream));
        void* args[4] = { &width, &height, &pixel_d, &blurredPixel_d };
        int requiredBlocks = (width * height + blockSize - 1) / blockSize;
        int usedBlocks = requiredBlocks < minGridSize ? requiredBlocks : minGridSize;
        // Grid: (usedBlocks, 1, 1)
        // Block: (blockSize, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_blur, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, dynamicSharedMemorySize, stream));
        CUDA_CHECK(cudaMemcpyAsync(blurredPixel_h, blurredPixel_d, sizeof(float) * width * height, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < width * height; i++) {
            cuda::std::div_t pixelPos = cuda::std::div(i, width);
            int pixelX = pixelPos.rem;
            int pixelY = pixelPos.quot;
            int arrayIndex = 0;
            float color = 0.0f;
            float sumOfCoefficients = 0.0f;
            // Applying a blur operation to the pixel at (pixelX, pixelY) by utilizing cuda::std::array for the offsets of neighboring pixels and the coefficients of the blur operation.
            for(auto coeff : blurCoefficients_h) { 
                int neighborX = pixelX + blurOffsetX_h[arrayIndex];
                int neighborY = pixelY + blurOffsetY_h[arrayIndex];
                if (neighborX >= 0 && neighborX < width && neighborY >= 0 && neighborY < height) {
                    color = cuda::std::fmaf(coeff, (*pixel_h)[neighborX + neighborY * width], color);
                }
                sumOfCoefficients += coeff;
                arrayIndex++;
            };
            assert(fabsf((*blurredPixel_h)[pixelX + pixelY * width] - color / sumOfCoefficients) < ERROR_TOLERANCE);
        }
    }
    // Freeing device memory.
    CUDA_CHECK(cudaFreeAsync(pixel_d, stream));
    CUDA_CHECK(cudaFreeAsync(blurredPixel_d, stream));
    // Freeing host memory.
    delete [] pixel_h;
    delete [] blurredPixel_h;
    delete [] testWidth;
    delete [] testHeight;
    delete [] testInput;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // Freeing other resources.
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    constexpr int WIDTH = 100;
    constexpr int HEIGHT = 100;
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 10000;

    int deviceId = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    size_t dynamicSharedMemorySize = 0;
    int minGridSize;
    int blockSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, (void*)k_blur, dynamicSharedMemorySize));

    cuda::std::array<float, MAX_PIXELS>* pixel_h = new cuda::std::array<float, MAX_PIXELS>();
    cuda::std::array<float, MAX_PIXELS>* blurredPixel_h = new cuda::std::array<float, MAX_PIXELS>();

    std::mt19937 generator(123);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        (*pixel_h)[i] = distribution(generator);
    }

    cuda::std::array<float, MAX_PIXELS>* pixel_d;
    cuda::std::array<float, MAX_PIXELS>* blurredPixel_d;
    CUDA_CHECK(cudaMallocAsync(&pixel_d, sizeof(cuda::std::array<float, MAX_PIXELS>), stream));
    CUDA_CHECK(cudaMallocAsync(&blurredPixel_d, sizeof(cuda::std::array<float, MAX_PIXELS>), stream));

    CUDA_CHECK(cudaMemcpyAsync(pixel_d, pixel_h, sizeof(float) * WIDTH * HEIGHT, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int width = WIDTH;
    int height = HEIGHT;
    void* args[4] = { &width, &height, &pixel_d, &blurredPixel_d };
    int requiredBlocks = (WIDTH * HEIGHT + blockSize - 1) / blockSize;
    int usedBlocks = requiredBlocks < minGridSize ? requiredBlocks : minGridSize;

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_blur, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, dynamicSharedMemorySize, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_blur, dim3(usedBlocks, 1, 1), dim3(blockSize, 1, 1), args, dynamicSharedMemorySize, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(pixel_d, stream));
    CUDA_CHECK(cudaFreeAsync(blurredPixel_d, stream));
    delete pixel_h;
    delete blurredPixel_h;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}