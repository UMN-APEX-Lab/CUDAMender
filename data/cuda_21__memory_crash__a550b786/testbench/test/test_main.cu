#include "stencil3d.h"
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
using namespace std;

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

void stencil3d(float *input, float *output, unsigned int N)
{
    float *input_d, *output_d;
    cudaMalloc(&input_d, N * N * N * sizeof(float));
    cudaMalloc(&output_d, N * N * N * sizeof(float));
    cudaCheckErrors("cudaMalloc failed");

    // Copy the memory from the host to the GPU
    cudaMemcpy(input_d, input, N * N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpu H2D failed");

    // Perform the 3d stencil operation
    dim3 numberOfThreadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 numberOfBlocks((N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM,
                        (N + BLOCK_DIM - 1) / BLOCK_DIM);
    stencil3d_kernel<<<numberOfBlocks, numberOfThreadsPerBlock>>>(input_d, output_d, N);
    cudaCheckErrors("kernel execution failed");

    // Copy the result back to the host
    cudaMemcpy(output, output_d, N * N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failed");

    // Free the GPU Memory
    cudaFree(input_d);
    cudaFree(output_d);
}

void test(unsigned int N)
{
    // Allocate host memory
    float *img = (float *)malloc(N * N * N * sizeof(float));
    float *out = (float *)malloc(N * N * N * sizeof(float));

    // Populate the arrays
    for (int i = 0; i < N * N * N; i++)
    {
        img[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Time the GPU operation
    stencil3d(img, out, N);

    // Free the allocated memory
    free(img);
    free(out);
}

void launch()
{
    cudaDeviceSynchronize();

    // Seed the random number generator
    srand(42);

    const unsigned int TESTS = 2;
    unsigned int Ns[]        = {1 << 6, 4096};
    for (int i = 0; i < TESTS; i++)
    {
        test(Ns[i]);
    }
}

int main() {
    launch();
}