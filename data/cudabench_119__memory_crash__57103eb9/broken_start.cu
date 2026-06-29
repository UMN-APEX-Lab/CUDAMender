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

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 8;

    __shared__ float sA[TILE_M][TILE_K];
    __shared__ float sB[TILE_K][TILE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE_K) {
        // Load tile from A
        if (row < M && (k + tx) < K)
            sA[ty][tx] = A[row * K + (k + tx)];
        else
            sA[ty][tx] = 0.0f;

        // Load tile from B
        if ((k + ty) < K && col < N)
            sB[ty][tx] = B[(k + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int tk = 0; tk < TILE_K; ++tk) {
            sum += sA[ty][tk] * sB[tk][tx];
        }

        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 262144;
    const int K = 4096;
    const int N = 2048;

    size_t sizeA = M * K;
    size_t sizeB = K * N;
    size_t sizeC = M * N;

    float* h_matA = new float[sizeA];
    float* h_matB = new float[sizeB];
    float* h_matC_out = new float[sizeC];

    read_binary("data/matA.bin", h_matA, sizeA);
    read_binary("data/matB.bin", h_matB, sizeB);

    float* d_matA = nullptr;
    float* d_matB = nullptr;
    float* d_matC_out = nullptr;

    cudaMalloc(&d_matA, sizeA * sizeof(float));
    cudaMalloc(&d_matB, sizeB * sizeof(float));
    cudaMalloc(&d_matC_out, sizeC * sizeof(float));

    cudaMemcpy(d_matA, h_matA, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, sizeB * sizeof(float), cudaMemcpyHostToDevice);

    const int TILE_M = 32;
    const int TILE_N = 32;
    dim3 block(TILE_N, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    matmul_kernel<<<grid, block>>>(d_matA, d_matB, d_matC_out, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_matC_out, d_matC_out, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    write_binary("data/matC_out.bin", h_matC_out, sizeC);

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC_out);
    delete[] h_matA;
    delete[] h_matB;
    delete[] h_matC_out;

    return 0;
}