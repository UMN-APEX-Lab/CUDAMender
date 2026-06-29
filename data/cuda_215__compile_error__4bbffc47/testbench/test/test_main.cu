#include "lbp.h"
#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

void launch() {
    const int NUM_TESTS = 7;
    const int WIDTH = 1024;
    const int HEIGHTS[NUM_TESTS] = {64, 128, 256, 512, 1024, 2048, 4096};
    const int BLOCK_SIZE_X = 32;
    const int BLOCK_SIZE_Y = 8;

    auto cpuLBP = [](const std::vector<unsigned char>& input, int x, int y, int width) {
        int idx = y * width + x;
        unsigned char c = input[idx];
        unsigned char lbpCode = 0;
        lbpCode |= (input[(y-1)*width + (x-1)] >= c) << BIT_TOP_LEFT;
        lbpCode |= (input[(y-1)*width + x    ] >= c) << BIT_TOP;
        lbpCode |= (input[(y-1)*width + (x+1)] >= c) << BIT_TOP_RIGHT;
        lbpCode |= (input[y*width + (x+1)]    >= c) << BIT_RIGHT;
        lbpCode |= (input[(y+1)*width + (x+1)] >= c) << BIT_BOTTOM_RIGHT;
        lbpCode |= (input[(y+1)*width + x    ] >= c) << BIT_BOTTOM;
        lbpCode |= (input[(y+1)*width + (x-1)] >= c) << BIT_BOTTOM_LEFT;
        lbpCode |= (input[y*width + (x-1)]    >= c) << BIT_LEFT;
        return lbpCode;
    };

    auto fillRandom = [](std::vector<unsigned char>& input) {
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> dist(0, 255);
        for (auto& val : input) val = static_cast<unsigned char>(dist(rng));
    };

    auto verify = [&](const std::vector<unsigned char>& input, const std::vector<unsigned char>& output, int width, int height) {
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                int idx = y * width + x;
                unsigned char expected = cpuLBP(input, x, y, width);
                assert(output[idx] == expected);
            }
        }
    };

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    const int maxSharedMem = prop.sharedMemPerBlock;
    const int numSMs = prop.multiProcessorCount;

    unsigned char *input_d, *output_d;
    int maxImgSize = WIDTH * HEIGHTS[NUM_TESTS - 1];
    CUDA_CHECK(cudaMallocAsync((void**)&input_d, maxImgSize * sizeof(unsigned char), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&output_d, maxImgSize * sizeof(unsigned char), stream));

    for (int test = 0; test < NUM_TESTS; ++test) {
        int height = HEIGHTS[test];
        int imgSize = WIDTH * height;

        std::vector<unsigned char> input_h(imgSize);
        std::vector<unsigned char> output_h(imgSize);

        fillRandom(input_h);

        CUDA_CHECK(cudaMemcpyAsync(input_d, input_h.data(), imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));

        dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
        size_t requestedSharedMem = (blockDim.x + HALO_PADDING) * (blockDim.y + HALO_PADDING) * sizeof(unsigned char);
        size_t sharedMem = std::min(requestedSharedMem, static_cast<size_t>(maxSharedMem));

        int gridX = std::min((int)((WIDTH + blockDim.x - 1) / blockDim.x), numSMs * maxBlocksPerSM);
        int gridY = std::min((int)((height + blockDim.y - 1) / blockDim.y), numSMs * maxBlocksPerSM);
        dim3 gridDim(gridX, gridY);

        void* args[] = { (void*)&input_d, (void*)&output_d, (void*)&WIDTH, (void*)&height };
        CUDA_CHECK(cudaLaunchKernel((void*)k_computeLBP, gridDim, blockDim, args, sharedMem, stream));

        CUDA_CHECK(cudaMemcpyAsync(output_h.data(), output_d, imgSize * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        verify(input_h, output_h, WIDTH, height);
    }

    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    const int WIDTH = 1024;
    const int HEIGHT = 8192;
    const int imgSize = WIDTH * HEIGHT;
    const int BLOCK_SIZE_X = 32;
    const int BLOCK_SIZE_Y = 8;
    const int WARMUP_ITERS = 3;
    const int TIMED_ITERS = 100;

    std::vector<unsigned char> input_h(imgSize);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& val : input_h) val = static_cast<unsigned char>(dist(rng));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int maxBlocksPerSM = prop.maxThreadsPerMultiProcessor / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
    const int maxSharedMem = prop.sharedMemPerBlock;
    const int numSMs = prop.multiProcessorCount;

    unsigned char *input_d, *output_d;
    CUDA_CHECK(cudaMallocAsync((void**)&input_d, imgSize * sizeof(unsigned char), stream));
    CUDA_CHECK(cudaMallocAsync((void**)&output_d, imgSize * sizeof(unsigned char), stream));
    CUDA_CHECK(cudaMemcpyAsync(input_d, input_h.data(), imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));

    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    size_t requestedSharedMem = (blockDim.x + HALO_PADDING) * (blockDim.y + HALO_PADDING) * sizeof(unsigned char);
    size_t sharedMem = std::min(requestedSharedMem, static_cast<size_t>(maxSharedMem));

    int gridX = std::min((int)((WIDTH + blockDim.x - 1) / blockDim.x), numSMs * maxBlocksPerSM);
    int gridY = std::min((int)((HEIGHT + blockDim.y - 1) / blockDim.y), numSMs * maxBlocksPerSM);
    dim3 gridDim(gridX, gridY);

    void* args[] = { (void*)&input_d, (void*)&output_d, (void*)&WIDTH, (void*)&HEIGHT };

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_computeLBP, gridDim, blockDim, args, sharedMem, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; ++i) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_computeLBP, gridDim, blockDim, args, sharedMem, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}