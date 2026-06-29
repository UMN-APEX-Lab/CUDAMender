#include "angular_momentum.h"
#include <cassert>
#include <nvtx3/nvToolsExt.h>

void launch() {
    const unsigned int NUM_TEST_CASES = 2;
    const unsigned int MAX_PARTICLE_COUNT = 4;
    const unsigned int BLOCK_SIZE = 256;
    const float TOL = 1e-4f;
    const float3 ZERO_VEC = {0.0f, 0.0f, 0.0f};

    cudaDeviceProp deviceProp;
    int currentDevice;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, currentDevice));

    int numSMs = deviceProp.multiProcessorCount;
    int maxBlocksPerSM = deviceProp.maxBlocksPerMultiProcessor;
    int numBlocks = numSMs * maxBlocksPerSM;

    auto vecDiff = [](const float3 &a, const float3 &b) -> float3 {
        return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    };
    
    auto vecNorm = [](const float3 &v) -> float {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    };

    unsigned int particleCountPerCase[NUM_TEST_CASES] = {2, 4};
    
    float mass_h[NUM_TEST_CASES][MAX_PARTICLE_COUNT] = {
        {6.305726e+00f, 3.704031e+00f},
        {6.305726e+00f, 3.704031e+00f, 3.452430e+00f, 2.186230e+00f}
    };

    float3 pos_h[NUM_TEST_CASES][MAX_PARTICLE_COUNT] = {
        {{-2.275078e+00f, -3.681967e+00f, 2.129892e+00f}, {-4.756183e+00f, 4.598780e+00f, -4.431842e+00f}},
        {{2.129892e+00f, -4.756183e+00f, 4.598780e+00f}, {-4.431842e+00f, -4.658658e+00f, 4.768967e+00f}, {3.881811e+00f, 4.415114e+00f, 3.964936e+00f}, {-3.184883e+00f, 4.143817e-01f, -2.354646e+00f}}
    };

    float3 vel_h[NUM_TEST_CASES][MAX_PARTICLE_COUNT] = {
        {{-9.317315e-01f, 9.537933e-01f, 7.763622e-01f}, {8.830227e-01f, 7.929872e-01f, -6.369766e-01f}},
        {{2.858950e-01f, -8.132153e-01f, 9.140813e-01f}, {2.353761e-01f, 5.988926e-01f, -9.769579e-01f}, {-3.992917e-01f, -5.506736e-01f, -5.077804e-01f}, {1.830518e-02f, 4.711822e-01f, -2.127832e-01f}}
    };

    float3 validTotalAM_h[NUM_TEST_CASES] = {
        {-2.866796e+01, -2.709299e+01, -6.432711e+01},
        {4.477622e+00, -1.610214e+01, -1.270816e+01}
    };

    for (unsigned int i = 0; i < NUM_TEST_CASES; ++i) {
        float *mass_d;
        float3 *pos_d, *vel_d, *totalAM_d;
        unsigned int particleCount = particleCountPerCase[i];

        float3 gpuTotalAM_h;

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        CUDA_CHECK(cudaMallocAsync(&mass_d, particleCount * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&pos_d, particleCount * sizeof(float3), stream));
        CUDA_CHECK(cudaMallocAsync(&vel_d, particleCount * sizeof(float3), stream));
        CUDA_CHECK(cudaMallocAsync(&totalAM_d, sizeof(float3), stream));

        CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h[i], particleCount * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(pos_d, pos_h[i], particleCount * sizeof(float3), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(vel_d, vel_h[i], particleCount * sizeof(float3), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(totalAM_d, &ZERO_VEC, sizeof(float3), cudaMemcpyHostToDevice, stream));

        void *args[] = {&mass_d, &pos_d, &vel_d, &totalAM_d, &particleCount};

        dim3 gridDim(numBlocks);
        dim3 blockDim(BLOCK_SIZE);

        CUDA_CHECK(cudaLaunchKernel((void*)k_computeAngularMomentum, gridDim, blockDim, args, 0, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(&gpuTotalAM_h, totalAM_d, sizeof(float3), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        assert(vecNorm(vecDiff(gpuTotalAM_h, validTotalAM_h[i])) < TOL);

        CUDA_CHECK(cudaFreeAsync(mass_d, stream));
        CUDA_CHECK(cudaFreeAsync(pos_d, stream));
        CUDA_CHECK(cudaFreeAsync(vel_d, stream));
        CUDA_CHECK(cudaFreeAsync(totalAM_d, stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

void benchmark() {
    const unsigned int particleCount = 2000000;
    const unsigned int BLOCK_SIZE = 256;
    const int WARMUP_ITERS = 3;
    const int TIMED_ITERS = 100;

    cudaDeviceProp deviceProp;
    int currentDevice;
    CUDA_CHECK(cudaGetDevice(&currentDevice));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, currentDevice));

    int numSMs = deviceProp.multiProcessorCount;
    int maxBlocksPerSM = deviceProp.maxBlocksPerMultiProcessor;
    int numBlocks = numSMs * maxBlocksPerSM;

    float *mass_h = (float *)malloc(particleCount * sizeof(float));
    float3 *pos_h = (float3 *)malloc(particleCount * sizeof(float3));
    float3 *vel_h = (float3 *)malloc(particleCount * sizeof(float3));
    for (unsigned int i = 0; i < particleCount; ++i) {
        mass_h[i] = 1.0f + (float)(i % 100) * 0.01f;
        pos_h[i] = make_float3((float)(i % 1000) * 0.1f, (float)((i + 1) % 1000) * 0.1f, (float)((i + 2) % 1000) * 0.1f);
        vel_h[i] = make_float3((float)(i % 500) * 0.01f, (float)((i + 3) % 500) * 0.01f, (float)((i + 7) % 500) * 0.01f);
    }

    float *mass_d;
    float3 *pos_d, *vel_d, *totalAM_d;
    const float3 ZERO_VEC = {0.0f, 0.0f, 0.0f};

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMallocAsync(&mass_d, particleCount * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&pos_d, particleCount * sizeof(float3), stream));
    CUDA_CHECK(cudaMallocAsync(&vel_d, particleCount * sizeof(float3), stream));
    CUDA_CHECK(cudaMallocAsync(&totalAM_d, sizeof(float3), stream));

    CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h, particleCount * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(pos_d, pos_h, particleCount * sizeof(float3), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(vel_d, vel_h, particleCount * sizeof(float3), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    void *args[] = {&mass_d, &pos_d, &vel_d, &totalAM_d, (void *)&particleCount};
    dim3 gridDim(numBlocks);
    dim3 blockDim(BLOCK_SIZE);

    for (int i = 0; i < WARMUP_ITERS; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(totalAM_d, &ZERO_VEC, sizeof(float3), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void *)k_computeAngularMomentum, gridDim, blockDim, args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(totalAM_d, &ZERO_VEC, sizeof(float3), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaLaunchKernel((void *)k_computeAngularMomentum, gridDim, blockDim, args, 0, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(mass_d, stream));
    CUDA_CHECK(cudaFreeAsync(pos_d, stream));
    CUDA_CHECK(cudaFreeAsync(vel_d, stream));
    CUDA_CHECK(cudaFreeAsync(totalAM_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    free(mass_h);
    free(pos_h);
    free(vel_h);
}