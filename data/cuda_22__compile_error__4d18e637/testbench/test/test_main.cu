#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <nvtx3/nvToolsExt.h>
#include "transpose.h"

#define cudaCheckErrors(msg)                                                                 \
    do                                                                                       \
    {                                                                                        \
        cudaError_t __err = cudaGetLastError();                                              \
        if (__err != cudaSuccess)                                                            \
        {                                                                                    \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)", msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__);                                                     \
            fprintf(stderr, "*** FAILED - ABORTING");                                        \
            exit(1);                                                                         \
        }                                                                                    \
    }                                                                                        \
    while (0)

bool validate_transpose(const float *input, const float *output, int width, int height)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            if (input[i * width + j] != output[j * height + i])
            {
                return false;
            }
        }
    }
    return true;
}

int launch()
{
    int width       = 1024;
    int height      = 768;
    int size        = width * height;
    float *h_input  = new float[size];
    float *h_output = new float[size];

    for (int i = 0; i < size; ++i)
    {
        h_input[i] = static_cast<float>(i);
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    transpose<<<grid_size, block_size>>>(d_input, d_output, width, height);
    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    assert(validate_transpose(h_input, h_output, width, height));

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

void benchmark()
{
    int width  = 4096;
    int height = 4096;
    int size   = width * height;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaCheckErrors("cudaMalloc failure");

    float *h_input = new float[size];
    for (int i = 0; i < size; ++i)
        h_input[i] = static_cast<float>(i);
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");
    delete[] h_input;

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int i = 0; i < 3; ++i)
    {
        transpose<<<grid_size, block_size>>>(d_input, d_output, width, height);
        cudaDeviceSynchronize();
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < 100; ++i)
    {
        transpose<<<grid_size, block_size>>>(d_input, d_output, width, height);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    cudaFree(d_input);
    cudaFree(d_output);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0)
    {
        benchmark();
    }
    else
    {
        launch();
    }
    return 0;
}