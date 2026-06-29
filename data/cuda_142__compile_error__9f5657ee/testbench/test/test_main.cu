#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <random>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include "nbody_kernel.h"

#define CUDA_CHECK(call) {                                      \
    cudaError_t error = call;                                  \
    if(error != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error at %s: %d - %s \n",        \
                __FILE__, __LINE__, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);                                    \
    }                                                          \
}

// Test-related constants
constexpr float EPSILON = 0.001f;
constexpr float ACCEPTABLE_MAGNITUDE = 1e12f;

void launch() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int threadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlocks = (prop.maxThreadsPerMultiProcessor / threadsPerBlock) * prop.multiProcessorCount;
    int requiredBlocks = NUM_PARTICLES / NUM_PARTICLES_PER_TILE;
    int usedBlocks = maxBlocks < requiredBlocks? maxBlocks : requiredBlocks;
    dim3 gridDim(usedBlocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    // Host buffer allocation.
    float *xPosition_h = new float[NUM_PARTICLES];
    float *yPosition_h = new float[NUM_PARTICLES];
    float *mass_h = new float[NUM_PARTICLES];
    float *xForce_h = new float[NUM_PARTICLES];
    float *yForce_h = new float[NUM_PARTICLES];

    // Device buffer allocation.
    float *xPosition_d;
    float *yPosition_d;
    float *mass_d;
    float *xForce_d;
    float *yForce_d;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocAsync(&xPosition_d, ALLOCATION_SIZE, stream));
    CUDA_CHECK(cudaMallocAsync(&yPosition_d, ALLOCATION_SIZE, stream));
    CUDA_CHECK(cudaMallocAsync(&mass_d, ALLOCATION_SIZE, stream));
    CUDA_CHECK(cudaMallocAsync(&xForce_d, ALLOCATION_SIZE, stream));
    CUDA_CHECK(cudaMallocAsync(&yForce_d, ALLOCATION_SIZE, stream));
    auto hToD = cudaMemcpyHostToDevice;
    auto dToH = cudaMemcpyDeviceToHost;
    void *args[5] = { &xPosition_d, &yPosition_d, &mass_d, &xForce_d, &yForce_d };
    
    // Initializing all host buffers.
    for(int i = 0; i < NUM_PARTICLES; i++) {
        xPosition_h[i] = (i % 100) * 10;
        yPosition_h[i] = (i / 100) * 10;
        mass_h[i] = 1.0f;
        xForce_h[i] = 0.0f;
        yForce_h[i] = 0.0f;
    }

    // Initializing all device buffers.
    CUDA_CHECK(cudaMemcpyAsync(xPosition_d, xPosition_h, ALLOCATION_SIZE, hToD, stream));
    CUDA_CHECK(cudaMemcpyAsync(yPosition_d, yPosition_h, ALLOCATION_SIZE, hToD, stream));
    CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h, ALLOCATION_SIZE, hToD, stream));
    CUDA_CHECK(cudaMemcpyAsync(xForce_d, xForce_h, ALLOCATION_SIZE, hToD, stream));
    CUDA_CHECK(cudaMemcpyAsync(yForce_d, yForce_h, ALLOCATION_SIZE, hToD, stream));

    // Test 1: All particles in the exact same position -> no force applied.
    {
        for(int i = 0; i < NUM_PARTICLES; i++) {
            xPosition_h[i] = 500.0f;
            yPosition_h[i] = 500.0f;
        }

        CUDA_CHECK(cudaMemcpyAsync(xPosition_d, xPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(yPosition_d, yPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h, ALLOCATION_SIZE, hToD, stream));
        
        // Grid: (2, 1, 1)
        // Block: (1024, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateNBodyForce, 
                                    gridDim, 
                                    blockDim, 
                                    args, 
                                    SHARED_MEM_ALLOCATION_SIZE, 
                                    stream));
        CUDA_CHECK(cudaMemcpyAsync(xForce_h, xForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaMemcpyAsync(yForce_h, yForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int i = 0; i < NUM_PARTICLES; i++) {
            assert(fabs(xForce_h[i]) < EPSILON);
            assert(fabs(yForce_h[i]) < EPSILON);
        }
    }

    // Test 2: All particles are on same line with with equal distances -> end points have equal magnitude of force, inner points have equal magnitude of force, middle point has zero force due to symmetry of all forces acting on it.
    {
        for(int i = 0; i < NUM_PARTICLES; i++) {
            xPosition_h[i] = 100.0f + i * 100.0f;
            yPosition_h[i] = 100.0f + i * 100.0f;
        }

        CUDA_CHECK(cudaMemcpyAsync(xPosition_d, xPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(yPosition_d, yPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h, ALLOCATION_SIZE, hToD, stream));
        
        // Grid: (2, 1, 1)
        // Block: (1024, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateNBodyForce, 
                                    gridDim, 
                                    blockDim, 
                                    args, 
                                    SHARED_MEM_ALLOCATION_SIZE, 
                                    stream));
        CUDA_CHECK(cudaMemcpyAsync(xForce_h, xForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaMemcpyAsync(yForce_h, yForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for(int i = 0; i < NUM_PARTICLES / 2; i++) {
            assert(fabs(fabs(xForce_h[i]) - fabs(xForce_h[NUM_PARTICLES - i - 1])) < EPSILON);
            assert(fabs(fabs(yForce_h[i]) - fabs(yForce_h[NUM_PARTICLES - i - 1])) < EPSILON);
        }
        
        assert(fabs(xForce_h[NUM_PARTICLES / 2]) < EPSILON);
        assert(fabs(yForce_h[NUM_PARTICLES / 2]) < EPSILON);        
    }

    // Test 3: Test 2: All particles are on the same line with equal distances -> the endpoints have equal magnitude of force, the inner points have equal magnitude of force, and the middle point has zero force due to the symmetry of all the forces acting on it.
    {
        float radius = 500.0f;
        float centerX = 500.0f;
        float centerY = 500.0f;

        for(int i = 0; i < NUM_PARTICLES; i++) {
            float pi = acos(-1);
            float posX = sin( (i / (float) NUM_PARTICLES) * 2 * pi) * radius;
            float posY = cos( (i / (float) NUM_PARTICLES) * 2 * pi) * radius;
            xPosition_h[i] = posX + centerX;
            yPosition_h[i] = posY + centerY;
        }

        CUDA_CHECK(cudaMemcpyAsync(xPosition_d, xPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(yPosition_d, yPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h, ALLOCATION_SIZE, hToD, stream));
        
        // Grid: (2, 1, 1)
        // Block: (1024, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateNBodyForce, 
                                    gridDim, 
                                    blockDim, 
                                    args, 
                                    SHARED_MEM_ALLOCATION_SIZE, 
                                    stream));
        CUDA_CHECK(cudaMemcpyAsync(xForce_h, xForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaMemcpyAsync(yForce_h, yForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for(int i = 0; i < NUM_PARTICLES - 1; i++) {
            float magnitude = sqrt(xForce_h[i] * xForce_h[i] + yForce_h[i] * yForce_h[i]);
            float unitVectorX = xForce_h[i] / magnitude;
            float unitVectorY = yForce_h[i] / magnitude;

            for(int j = i + 1; j < NUM_PARTICLES; j++) {
                float magnitude2 = sqrt(xForce_h[j] * xForce_h[j] + yForce_h[j] * yForce_h[j]);
                float unitVector2X = xForce_h[j] / magnitude2;
                float unitVector2Y = yForce_h[j] / magnitude2;
                assert(fabs(magnitude - magnitude2) < EPSILON);
            }

            float targetVectorX = radius * unitVectorX;
            float targetVectorY = radius * unitVectorY;
            float targetPointX = targetVectorX + xPosition_h[i];
            float targetPointY = targetVectorY + yPosition_h[i];

            // Checking whether forces are directed towards the center of the circle.
            assert(fabs(targetPointX - centerX) < EPSILON);
            assert(fabs(targetPointY - centerY) < EPSILON);
        }
    }

    // Test 4: Randomly scattered particles -> the result is compared with the host-side calculation.
    {
        std::mt19937 generator(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(0.0f, 1000.0f);

        for(int i = 0; i < NUM_PARTICLES; i++) {
            xPosition_h[i] = dist(generator);
            yPosition_h[i] = dist(generator);
        }

        CUDA_CHECK(cudaMemcpyAsync(xPosition_d, xPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(yPosition_d, yPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h, ALLOCATION_SIZE, hToD, stream));

        // Grid: (2, 1, 1)
        // Block: (1024, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateNBodyForce,
                                    gridDim,
                                    blockDim,
                                    args,
                                    SHARED_MEM_ALLOCATION_SIZE,
                                    stream));
        CUDA_CHECK(cudaMemcpyAsync(xForce_h, xForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaMemcpyAsync(yForce_h, yForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for(int i = 0; i < NUM_PARTICLES; i++) {
            float xForce = 0.0f;
            float yForce = 0.0f;

            for(int j = 0; j < NUM_PARTICLES; j++) {
                float dx = xPosition_h[i] - xPosition_h[j];
                float dy = yPosition_h[i] - yPosition_h[j];
                float distanceSquared = dx * dx + dy * dy;
                float inverseDistance = 1.0f / sqrtf(distanceSquared + SMOOTHING);
                float forceMultiplier = distanceSquared < VERY_CLOSE ?
                                        0.0f :
                                        -GRAVITATIONAL_FORCE_CONSTANT * mass_h[i] * mass_h[j] *
                                        inverseDistance * inverseDistance * inverseDistance;
                xForce = fmaf(dx, forceMultiplier, xForce);
                yForce = fmaf(dy, forceMultiplier, yForce);
            }

            assert(fabs(xForce - xForce_h[i]) < EPSILON);
            assert(fabs(yForce - yForce_h[i]) < EPSILON);
        }
    }

    // Test 5: Very close particles -> The smoothing variable prevents divide-by-zero errors, avoids denormal values, and keeps the force magnitude within acceptable limits.
    {
        for(int i = 0; i < NUM_PARTICLES; i++) {
            xPosition_h[i] = i * 0.00001f;
            yPosition_h[i] = i * 0.00001f;
        }

        CUDA_CHECK(cudaMemcpyAsync(xPosition_d, xPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(yPosition_d, yPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h, ALLOCATION_SIZE, hToD, stream));

        // Grid: (2, 1, 1)
        // Block: (1024, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateNBodyForce,
                                    gridDim,
                                    blockDim,
                                    args,
                                    SHARED_MEM_ALLOCATION_SIZE,
                                    stream));
        CUDA_CHECK(cudaMemcpyAsync(xForce_h, xForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaMemcpyAsync(yForce_h, yForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for(int i = 0; i < NUM_PARTICLES; i++) {
            float xForce = 0.0f;
            float yForce = 0.0f;

            for(int j = 0; j < NUM_PARTICLES; j++) {
                float dx = xPosition_h[i] - xPosition_h[j];
                float dy = yPosition_h[i] - yPosition_h[j];
                float massMultiplied = mass_h[i] * mass_h[j];
                float distanceSquared = dx * dx + dy * dy;
                float inverseDistance = 1.0f / sqrtf(distanceSquared + SMOOTHING);
                float forceMultiplier = distanceSquared < VERY_CLOSE ?
                                        0.0f :
                                        -GRAVITATIONAL_FORCE_CONSTANT * massMultiplied *
                                        inverseDistance * inverseDistance * inverseDistance;
                xForce = fmaf(dx, forceMultiplier, xForce);
                yForce = fmaf(dy, forceMultiplier, yForce);
            }

            assert(fabs(xForce - xForce_h[i]) < EPSILON);
            assert(fabs(yForce - yForce_h[i]) < EPSILON);
            assert(!isnan(xForce));
            assert(!isnan(yForce));
            assert(!isinf(xForce));
            assert(!isinf(yForce));
            assert(fabs(xForce_h[i]) < ACCEPTABLE_MAGNITUDE);
            assert(fabs(yForce_h[i]) < ACCEPTABLE_MAGNITUDE);
        }
    }

    // Test 6: One of the particles is at infinity  -> all particles have nan force.
    {
        for(int i = 0; i < NUM_PARTICLES; i++) {
            xPosition_h[i] = ((i == 0) ? INFINITY : i);
            yPosition_h[i] = ((i == 0) ? INFINITY : i);
        }

        CUDA_CHECK(cudaMemcpyAsync(xPosition_d, xPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(yPosition_d, yPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h, ALLOCATION_SIZE, hToD, stream));

        // Grid: (2, 1, 1)
        // Block: (1024, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateNBodyForce,
                                    gridDim,
                                    blockDim,
                                    args,
                                    SHARED_MEM_ALLOCATION_SIZE,
                                    stream));
        CUDA_CHECK(cudaMemcpyAsync(xForce_h, xForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaMemcpyAsync(yForce_h, yForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for(int i = 0; i < NUM_PARTICLES; i++) {
            assert(isnan(xForce_h[i]));
            assert(isnan(yForce_h[i]));
        }
    }

    // Test 7: Half of the particles are CLOSE_TO_INFINITE from the center, and the other half are CLOSE_TO_INFINITE from the center in the opposite direction -> only the x-component of the forces has nan value.
    {
        float CLOSE_TO_INFINITE = 3.0e+38f;

        for(int i = 0; i < NUM_PARTICLES; i++) {
            xPosition_h[i] = ((i < NUM_PARTICLES / 2) ? -CLOSE_TO_INFINITE : CLOSE_TO_INFINITE);
            yPosition_h[i] = i;
        }

        CUDA_CHECK(cudaMemcpyAsync(xPosition_d, xPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(yPosition_d, yPosition_h, ALLOCATION_SIZE, hToD, stream));
        CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h, ALLOCATION_SIZE, hToD, stream));

        // Grid: (2, 1, 1)
        // Block: (1024, 1, 1)
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateNBodyForce,
                                    gridDim,
                                    blockDim,
                                    args,
                                    SHARED_MEM_ALLOCATION_SIZE,
                                    stream));
        CUDA_CHECK(cudaMemcpyAsync(xForce_h, xForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaMemcpyAsync(yForce_h, yForce_d, ALLOCATION_SIZE, dToH, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        for(int i = 0; i < NUM_PARTICLES; i++) {
            assert(isnan(xForce_h[i]));
            assert(!isnan(yForce_h[i]));
        }
    }

    CUDA_CHECK(cudaFreeAsync(xPosition_d, stream));
    CUDA_CHECK(cudaFreeAsync(yPosition_d, stream));
    CUDA_CHECK(cudaFreeAsync(mass_d, stream));
    CUDA_CHECK(cudaFreeAsync(xForce_d, stream));
    CUDA_CHECK(cudaFreeAsync(yForce_d, stream));

    delete [] xPosition_h;
    delete [] yPosition_h;
    delete [] mass_h;
    delete [] xForce_h;
    delete [] yForce_h;

    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int threadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlocks = (prop.maxThreadsPerMultiProcessor / threadsPerBlock) * prop.multiProcessorCount;
    int requiredBlocks = NUM_PARTICLES / NUM_PARTICLES_PER_TILE;
    int usedBlocks = maxBlocks < requiredBlocks ? maxBlocks : requiredBlocks;
    dim3 gridDim(usedBlocks, 1, 1);
    dim3 blockDim(threadsPerBlock, 1, 1);

    float *xPosition_h = new float[NUM_PARTICLES];
    float *yPosition_h = new float[NUM_PARTICLES];
    float *mass_h = new float[NUM_PARTICLES];

    std::mt19937 generator(123);
    std::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        xPosition_h[i] = dist(generator);
        yPosition_h[i] = dist(generator);
        mass_h[i] = dist(generator) * 0.01f;
    }

    float *xPosition_d, *yPosition_d, *mass_d, *xForce_d, *yForce_d;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMallocAsync(&xPosition_d, ALLOCATION_SIZE, stream));
    CUDA_CHECK(cudaMallocAsync(&yPosition_d, ALLOCATION_SIZE, stream));
    CUDA_CHECK(cudaMallocAsync(&mass_d, ALLOCATION_SIZE, stream));
    CUDA_CHECK(cudaMallocAsync(&xForce_d, ALLOCATION_SIZE, stream));
    CUDA_CHECK(cudaMallocAsync(&yForce_d, ALLOCATION_SIZE, stream));

    CUDA_CHECK(cudaMemcpyAsync(xPosition_d, xPosition_h, ALLOCATION_SIZE, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(yPosition_d, yPosition_h, ALLOCATION_SIZE, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(mass_d, mass_h, ALLOCATION_SIZE, cudaMemcpyHostToDevice, stream));

    void *args[5] = { &xPosition_d, &yPosition_d, &mass_d, &xForce_d, &yForce_d };

    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 5000;

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateNBodyForce,
                                    gridDim, blockDim, args,
                                    SHARED_MEM_ALLOCATION_SIZE, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED_ITERS; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_calculateNBodyForce,
                                    gridDim, blockDim, args,
                                    SHARED_MEM_ALLOCATION_SIZE, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(xPosition_d, stream));
    CUDA_CHECK(cudaFreeAsync(yPosition_d, stream));
    CUDA_CHECK(cudaFreeAsync(mass_d, stream));
    CUDA_CHECK(cudaFreeAsync(xForce_d, stream));
    CUDA_CHECK(cudaFreeAsync(yForce_d, stream));

    delete[] xPosition_h;
    delete[] yPosition_h;
    delete[] mass_h;

    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}