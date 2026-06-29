#include "kernel_wrapper.h"
#include <stdio.h>
#include <vector>
#include <numeric>
#include <stdlib.h>
#include <string.h>
#include <nvtx3/nvToolsExt.h>

static void run_benchmark() {
    const int blockSize = 1024;
    const int warmup_iters = 3;
    const int timed_iters = 100000;

    std::vector<float> input(blockSize);
    for (int i = 0; i < blockSize; i++) {
        input[i] = static_cast<float>(i % 37) * 0.5f;
    }

    float *d_output, *d_input;
    cudaMalloc(&d_output, sizeof(float));
    cudaMalloc(&d_input, blockSize * sizeof(float));
    cudaMemcpy(d_input, input.data(), blockSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(blockSize);
    dim3 numBlocks(1);
    cudaLaunchConfig_t config = {0};
    config.gridDim = numBlocks;
    config.blockDim = threadsPerBlock;
    config.dynamicSmemBytes = ((config.blockDim.x + 31) / 32) * sizeof(float);

    for (int i = 0; i < warmup_iters; i++) {
        cudaLaunchKernelEx(&config, kernel, d_output, d_input);
    }
    cudaDeviceSynchronize();

    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed_iters; i++) {
        cudaLaunchKernelEx(&config, kernel, d_output, d_input);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        run_benchmark();
        return 0;
    }
    printf("=== CUDA-26 Warp Reduce Test Suite ===\n\n");
    
    // Test 1: All ones for various sizes
    printf("Test Category 1: All Ones\n");
    for (int size : {32, 64, 128, 256, 512, 1024}) {
        std::vector<float> input(size, 1.0f);
        float expected = static_cast<float>(size);
        char name[100];
        snprintf(name, sizeof(name), "AllOnes_Size%d", size);
        launch_with_input(size, input, expected, name);
    }
    printf("\n");
    
    // Test 2: All zeros
    printf("Test Category 2: All Zeros\n");
    for (int size : {32, 64, 128, 256, 512, 1024}) {
        std::vector<float> input(size, 0.0f);
        char name[100];
        snprintf(name, sizeof(name), "AllZeros_Size%d", size);
        launch_with_input(size, input, 0.0f, name);
    }
    printf("\n");
    
    // Test 3: Sequential values (0, 1, 2, 3, ...)
    printf("Test Category 3: Sequential Values\n");
    for (int size : {32, 64, 128, 256, 512, 1024}) {
        std::vector<float> input(size);
        for (int i = 0; i < size; i++) {
            input[i] = static_cast<float>(i);
        }
        float expected = static_cast<float>(size * (size - 1) / 2);
        char name[100];
        snprintf(name, sizeof(name), "Sequential_Size%d", size);
        launch_with_input(size, input, expected, name);
    }
    printf("\n");
    
    // Test 4: Negative values
    printf("Test Category 4: Negative Values\n");
    for (int size : {32, 64, 128, 256, 512, 1024}) {
        std::vector<float> input(size, -1.5f);
        float expected = -1.5f * size;
        char name[100];
        snprintf(name, sizeof(name), "AllNegative_Size%d", size);
        launch_with_input(size, input, expected, name);
    }
    printf("\n");
    
    // Test 5: Mixed positive and negative
    printf("Test Category 5: Mixed Positive and Negative\n");
    for (int size : {32, 64, 128, 256, 512, 1024}) {
        std::vector<float> input(size);
        for (int i = 0; i < size; i++) {
            input[i] = (i % 2 == 0) ? 1.0f : -1.0f;
        }
        float expected = (size % 2 == 0) ? 0.0f : 1.0f;
        char name[100];
        snprintf(name, sizeof(name), "Alternating_Size%d", size);
        launch_with_input(size, input, expected, name);
    }
    printf("\n");
    
    // Test 6: Large values
    printf("Test Category 6: Large Values\n");
    for (int size : {32, 64, 128, 256, 512, 1024}) {
        std::vector<float> input(size, 1000.0f);
        float expected = 1000.0f * size;
        char name[100];
        snprintf(name, sizeof(name), "LargeValues_Size%d", size);
        launch_with_input(size, input, expected, name);
    }
    printf("\n");
    
    // Test 7: Small fractional values
    printf("Test Category 7: Fractional Values\n");
    for (int size : {32, 64, 128, 256, 512, 1024}) {
        std::vector<float> input(size, 0.125f);
        float expected = 0.125f * size;
        char name[100];
        snprintf(name, sizeof(name), "Fractional_Size%d", size);
        launch_with_input(size, input, expected, name);
    }
    printf("\n");
    
    // Test 8: Single large value, rest zeros
    printf("Test Category 8: Single Large Value\n");
    for (int size : {32, 64, 128, 256, 512, 1024}) {
        std::vector<float> input(size, 0.0f);
        input[0] = 100.0f;
        char name[100];
        snprintf(name, sizeof(name), "SingleLarge_Size%d", size);
        launch_with_input(size, input, 100.0f, name);
    }
    printf("\n");
    
    // Test 9: Random pattern with fixed seed
    printf("Test Category 9: Random Pattern\n");
    srand(42);  // Set random seed for reproducibility
    
    for (int size : {32, 64, 128, 256, 512, 1024}) {
        std::vector<float> input(size);
        float expected = 0.0f;
        for (int i = 0; i < size; i++) {
            // Generate random float in range [-10.0, 10.0]
            input[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
            expected += input[i];
        }
        char name[100];
        snprintf(name, sizeof(name), "Random_Size%d", size);
        launch_with_input(size, input, expected, name);
    }
    printf("\n");
    
    // Test 10: Power of 2 sizes with specific patterns
    printf("Test Category 10: Power of 2 Boundary Tests\n");
    for (int size : {64, 128, 256, 512}) {
        std::vector<float> input(size);
        for (int i = 0; i < size; i++) {
            input[i] = (i % 3 == 0) ? 2.0f : -1.0f;
        }
        float expected = 0.0f;
        for (int i = 0; i < size; i++) {
            expected += input[i];
        }
        char name[100];
        snprintf(name, sizeof(name), "PowerOf2_Size%d", size);
        launch_with_input(size, input, expected, name);
    }
    printf("\n");
    
    // Test 11: Non-power of 2 sizes
    printf("Test Category 11: Non-Power of 2 Sizes\n");
    for (int size : {33, 100, 500, 777}) {
        std::vector<float> input(size, 2.5f);
        float expected = 2.5f * size;
        char name[100];
        snprintf(name, sizeof(name), "NonPowerOf2_Size%d", size);
        launch_with_input(size, input, expected, name);
    }
    printf("\n");
    
    // Test 12: Very small sizes
    printf("Test Category 12: Very Small Sizes\n");
    for (int size : {1, 2, 3, 7, 15, 31}) {
        std::vector<float> input(size, 5.0f);
        float expected = 5.0f * size;
        char name[100];
        snprintf(name, sizeof(name), "SmallSize_%d", size);
        launch_with_input(size, input, expected, name);
    }
    printf("\n");
    
    printf("=== All tests passed! ===\n");
    return 0;
}