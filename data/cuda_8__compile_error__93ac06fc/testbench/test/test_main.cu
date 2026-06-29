#include "cuda_graph.h"
#include "kernels.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

int launch()
{
    const int width  = 256;
    const int height = 256;
    const int size   = width * height;
    float* h_img     = (float*)malloc(size * sizeof(float));
    float* h_result  = (float*)malloc(size * sizeof(float));

    // Initialize the image with random values for testing
    std::srand(std::time(0));
    for (int i = 0; i < size; ++i)
    {
        h_img[i] = static_cast<float>(std::rand() % 256);
    }

    float* d_img;
    cudaMalloc(&d_img, size * sizeof(float));
    cudaMemcpy(d_img, h_img, size * sizeof(float), cudaMemcpyHostToDevice);

    // Run the CUDA graph
    run_cuda_graph(d_img, h_result, width, height);

    // Check the results using assertions
    for (int i = 0; i < size; ++i)
    {
        float expected = 2 * ((1.5f * h_img[i] / 255.0f) * 0.8f + (1.5f * h_img[i] / 255.0f));
        assert(h_result[i] == expected && "Assertion failed!");
    }

    free(h_img);
    free(h_result);
    cudaFree(d_img);
    return 0;
}

int benchmark()
{
    const int width  = 2048;
    const int height = 2048;
    const int size   = width * height;
    float* h_img     = (float*)malloc(size * sizeof(float));
    float* h_result  = (float*)malloc(size * sizeof(float));

    std::srand(42);
    for (int i = 0; i < size; ++i)
    {
        h_img[i] = static_cast<float>(std::rand() % 256);
    }

    float* d_img;
    cudaMalloc(&d_img, size * sizeof(float));
    cudaMemcpy(d_img, h_img, size * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup: 3 iterations
    for (int w = 0; w < 3; ++w)
    {
        cudaMemcpy(d_img, h_img, size * sizeof(float), cudaMemcpyHostToDevice);
        run_cuda_graph(d_img, h_result, width, height);
    }
    cudaDeviceSynchronize();

    // Timed region: 100 iterations
    nvtxRangePushA("bench_region");
    for (int t = 0; t < 100; ++t)
    {
        cudaMemcpy(d_img, h_img, size * sizeof(float), cudaMemcpyHostToDevice);
        run_cuda_graph(d_img, h_result, width, height);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    free(h_img);
    free(h_result);
    cudaFree(d_img);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--perf") == 0)
    {
        return benchmark();
    }
    return launch();
}