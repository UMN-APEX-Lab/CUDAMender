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

#define BLOCK_SIZE 32
#define TILE_SIZE 8

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;

    __shared__ float As[BLOCK_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][BLOCK_SIZE];

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int ph = 0; ph < num_tiles; ++ph) {
        int a_col = ph * TILE_SIZE + tx;
        int b_row = ph * TILE_SIZE + ty;

        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (col < N && b_row < K) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 16384;
    const int K = 4096;
    const int N = 2048;

    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    float* h_matA = new float[size_A];
    float* h_matB = new float[size_B];
    float* h_matC_out = new float[size_C];

    read_binary("data/matA.bin", h_matA, size_A);
    read_binary("data/matB.bin", h_matB, size_B);

    float* d_matA = nullptr;
    float* d_matB = nullptr;
    float* d_matC_out = nullptr;

    cudaMalloc(&d_matA, size_A * sizeof(float));
    cudaMalloc(&d_matB, size_B * sizeof(float));
    cudaMalloc(&d_matC_out, size_C * sizeof(float));

    cudaMemcpy(d_matA, h_matA, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, size_B * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<grid, block>>>(d_matA, d_matB, d_matC_out, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_matC_out, d_matC_out, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    write_binary("data/matC_out.bin", h_matC_out, size_C);

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC_out);
    delete[] h_matA;
    delete[] h_matB;
    delete[] h_matC_out;

    return 0;
}