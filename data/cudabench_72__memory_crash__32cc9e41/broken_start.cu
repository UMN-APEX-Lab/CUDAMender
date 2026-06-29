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

#define BATCH 16
#define M 512
#define K 256
#define N 128
#define TILE_M 32
#define TILE_N 32
#define TILE_K 8

__global__ void batched_matmul_kernel(const float* A, const float* B, float* C) {
    int b = blockIdx.z;
    int m = blockIdx.y * TILE_M + threadIdx.y;
    int n = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_K][TILE_N];

    float sum = 0.0f;

    for (int t = 0; t < K; t += TILE_K) {
        if (m < M && (t + threadIdx.x) < K) {
            sA[threadIdx.y][threadIdx.x] = A[b * M * K + m * K + t + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((t + threadIdx.y) < K && n < N) {
            sB[threadIdx.y][threadIdx.x] = B[b * K * N + (t + threadIdx.y) * N + n];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (m < M && n < N) {
        C[b * M * N + m * N + n] = sum;
    }
}

int main() {
    const size_t sizeA = BATCH * M * K;
    const size_t sizeB = BATCH * K * N;
    const size_t sizeC = BATCH * M * N;

    float* h_batchA = new float[sizeA];
    float* h_batchB = new float[sizeB];
    float* h_batchC_out = new float[sizeC];

    read_binary("data/batchA.bin", h_batchA, sizeA);
    read_binary("data/batchB.bin", h_batchB, sizeB);

    float* d_batchA = nullptr;
    float* d_batchB = nullptr;
    float* d_batchC_out = nullptr;

    cudaMalloc(&d_batchA, sizeA * sizeof(float));
    cudaMalloc(&d_batchB, sizeB * sizeof(float));
    cudaMalloc(&d_batchC_out, sizeC * sizeof(float));

    cudaMemcpy(d_batchA, h_batchA, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batchB, h_batchB, sizeB * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, BATCH);

    batched_matmul_kernel<<<grid, block>>>(d_batchA, d_batchB, d_batchC_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_batchC_out, d_batchC_out, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    write_binary("data/batchC_out.bin", h_batchC_out, sizeC);

    cudaFree(d_batchA);
    cudaFree(d_batchB);
    cudaFree(d_batchC_out);

    delete[] h_batchA;
    delete[] h_batchB;
    delete[] h_batchC_out;

    return 0;
}