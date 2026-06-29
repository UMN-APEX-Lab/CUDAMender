#include "stencil.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

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

bool validate(const int *input, const int *output, int size)
{
    for (int i = 0; i < size; ++i)
    {
        int expected = 0;
        for (int j = -RADIUS; j <= RADIUS; ++j)
        {
            int index = i + j;
            if (index >= 0 && index < size)
            {
                expected += input[index];
            }
        }
        if (output[i] != expected)
        {
            return false;
        }
    }
    return true;
}

int launch()
{
    int size = 1 << 24;   // 16M elements

    int *h_input  = new int[size];
    int *h_output = new int[size];

    for (int i = 0; i < size; ++i)
    {
        h_input[i] = rand() % 100;
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    stencil_1d<<<gridSize, BLOCK_SIZE>>>(d_input, d_output);
    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    assert(validate(h_input, h_output, size));

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

int benchmark()
{
    int size = 1 << 26;   // 64M elements

    int *h_input = new int[size];
    for (int i = 0; i < size; ++i)
    {
        h_input[i] = rand() % 100;
    }

    int *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(int));
    cudaMalloc(&d_output, size * sizeof(int));
    cudaCheckErrors("cudaMalloc failure");

    cudaMemcpy(d_input, h_input, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Warmup iterations (outside NVTX region)
    for (int i = 0; i < 3; ++i)
    {
        stencil_1d<<<gridSize, BLOCK_SIZE>>>(d_input, d_output);
        cudaDeviceSynchronize();
    }

    // Timed iterations
    nvtxRangePushA("bench_region");
    for (int i = 0; i < 100; ++i)
    {
        stencil_1d<<<gridSize, BLOCK_SIZE>>>(d_input, d_output);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();

    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0)
    {
        return benchmark();
    }
    return launch();
}