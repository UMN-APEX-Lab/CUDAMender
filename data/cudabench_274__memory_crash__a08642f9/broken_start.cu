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

#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 4

__global__ void conv_transposed_2d_kernel(const float* matA, const float* matB, float* matC_out, int M, int K, int N) {
    __shared__ float sA[BLOCK_M][BLOCK_K];
    __shared__ float sB[BLOCK_N][BLOCK_K];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * BLOCK_M + threadIdx.y;
    int col = bx * BLOCK_N + threadIdx.x;

    float sum = 0.0f;

    for (int k = 0; k < K; k += BLOCK_K) {
        int a_row = row;
        int a_col = k + threadIdx.x;
        if (a_row < M && a_col < K) {
            sA[threadIdx.y][threadIdx.x] = matA[a_row * K + a_col];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        int b_row = k + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N) {
            sB[threadIdx.x][threadIdx.y] = matB[b_row * N + b_col];
        } else {
            sB[threadIdx.x][threadIdx.y] = 0.0f;
        }
        
        __syncthreads();

        for (int i = 0; i < BLOCK_K; ++i) {
            sum += sA[threadIdx.y][i] * sB[threadIdx.x][i];
        }
        
        __syncthreads();
    }

    if (row < M && col < N) {
        matC_out[row * N + col] = sum;
    }
}

int main() {
    const int M = 147456;
    const int K = 64;
    const int N = 256;

    float* h_matA = new float[M * K];
    float* h_matB = new float[K * N];
    float* h_matC_out = new float[M * N];

    read_binary("data/matA.bin", h_matA, M * K);
    read_binary("data/matB.bin", h_matB, K * N);

    float* d_matA, *d_matB, *d_matC_out;
    cudaMalloc(&d_matA, M * K * sizeof(float));
    cudaMalloc(&d_matB, K * N * sizeof(float));
    cudaMalloc(&d_matC_out, M * N * sizeof(float));

    cudaMemcpy(d_matA, h_matA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    conv_transposed_2d_kernel<<<grid, block>>>(d_matA, d_matB, d_matC_out, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_matC_out, d_matC_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    write_binary("data/matC_out.bin", h_matC_out, M * N);

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC_out);
    delete[] h_matA;
    delete[] h_matB;
    delete[] h_matC_out;

    return 0;
}