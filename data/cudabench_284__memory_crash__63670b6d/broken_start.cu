// --- BEGIN REQUIRED BOILERPLATE ---
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cuda_runtime.h>

void read_binary(const std::string& filename, float* data, size_t size) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Can not open: " << filename << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(data), size * sizeof(float));
    in.close();
}

void write_binary(const std::string& filename, const float* data, size_t size) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Can not write: " << filename << std::endl;
        exit(1);
    }
    out.write(reinterpret_cast<const char*>(data), size * sizeof(float));
    out.close();
}
// --- END REQUIRED BOILERPLATE ---

#define INPUT_H 2048
#define INPUT_W 2048
#define KERNEL_H 24
#define KERNEL_W 24
#define OUTPUT_H 2025
#define OUTPUT_W 2025
#define BLOCK_SIZE 16

__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_h, int input_w,
    int kernel_h, int kernel_w,
    int output_h, int output_w
) {
    __shared__ float s_input[BLOCK_SIZE + KERNEL_H - 1][BLOCK_SIZE + KERNEL_W - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int gx = bx * BLOCK_SIZE + tx;
    int gy = by * BLOCK_SIZE + ty;

    int ix = bx * BLOCK_SIZE + tx - KERNEL_W + 1;
    int iy = by * BLOCK_SIZE + ty - KERNEL_H + 1;

    if (ix >= 0 && ix < input_w && iy >= 0 && iy < input_h) {
        s_input[ty + KERNEL_H - 1][tx + KERNEL_W - 1] = input[iy * input_w + ix];
    } else {
        s_input[ty + KERNEL_H - 1][tx + KERNEL_W - 1] = 0.0f;
    }

    __syncthreads();

    float sum = 0.0f;
    for (int ky = 0; ky < kernel_h; ky++) {
        for (int kx = 0; kx < kernel_w; kx++) {
            sum += s_input[ty + ky + KERNEL_H - 1][tx + kx + KERNEL_W - 1] * kernel[ky * kernel_w + kx];
        }
    }

    if (gx < output_w && gy < output_h) {
        output[gy * output_w + gx] = sum;
    }
}

int main() {
    const size_t input_size = INPUT_H * INPUT_W;
    const size_t kernel_size = KERNEL_H * KERNEL_W;
    const size_t output_size = OUTPUT_H * OUTPUT_W;

    float* h_input = new float[input_size];
    float* h_kernel = new float[kernel_size];
    float* h_output = new float[output_size];

    read_binary("data/conv_input.bin", h_input, input_size);
    read_binary("data/conv_kernel.bin", h_kernel, kernel_size);

    float* d_input = nullptr;
    float* d_kernel = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((OUTPUT_W + BLOCK_SIZE - 1) / BLOCK_SIZE, (OUTPUT_H + BLOCK_SIZE - 1) / BLOCK_SIZE);

    conv2d_kernel<<<grid, block>>>(
        d_input, d_kernel, d_output,
        INPUT_H, INPUT_W,
        KERNEL_H, KERNEL_W,
        OUTPUT_H, OUTPUT_W
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    write_binary("data/conv_output.bin", h_output, output_size);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;

    return 0;
}