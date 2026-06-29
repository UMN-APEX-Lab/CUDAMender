#include "k_nearest_neighbors.h"
#include "cuda_helpers.h"
#include <limits.h>
#include <float.h>
#undef NDEBUG
#include <assert.h>
#include <cstring>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

void launch() {
    const int TEST_CASES = 9;
    const int NUMEL_A = 10;
    const int NUMEL_B = 10;
    
    // Variable allocations
    int testCaseCount = TEST_CASES; // Number of test cases
    int numberOfPointsA = NUMEL_A;
    int numberOfPointsB = NUMEL_B;
    int outputIndices[TEST_CASES][NUMEL_A] = { { 4,4,9,4,0,2,7,7,2,7 },{6,2,5,7,2,5,2,7,3,3},{9,4,9,1,4,1,9,7,2,4},
        {0,6,2,1,7,2,9,9,2,2},{8,1,5,3,1,4,9,5,3,1},{0,2,4,4,3,0,7,5,6,4},{8,6,5,6,6,2,2,9,4,5},{0,5,3,6,0,4,7,6,1,4},{3,1,9,8,5,4,7,0,6,0} };

    // Test-cases
    float inputVectorA_h[TEST_CASES][NUMEL_A][N_DIMS] = {
        { {-0.38,-2.13,0.69},{-5.7,-4.5,-0.95},{9.23,7.69,9.65},{-6.54,2.21,4.76},{-7.71,-7.99,8.89},{7.93,-6.5,0.11},{7.06,6.76,-8.3},{4.66,9.78,-9.06},{2.19,-5.55,-2.43},{2.74,-0.14,-7.55}},
        { {7.84,7.03,-9.84},{-9.44,-6.52,6.45},{2.12,0.52,-4.83},{-7.45,2.44,0.11},{-5.57,-9.12,-4.06},{2.29,1.76,-5.07},{-0.76,-8.89,-5.96},{-5.9,-0.69,-4.59},{9.59,1.21,0.22},{2.45,0.02,-2.54}},
        { {-0.58,2.59,-0.67},{-9.12,-6.22,4.25},{-8.42,7.6,2.66},{6.51,6.53,-7.43},{-7.46,-9.41,4.89},{3.84,5.8,0.96},{-5.66,7.97,-0.96},{6.59,-6.21,3.64},{6.01,-0.08,-1.41},{-2.13,-5.11,4.1}},
        { {-9.23,-2.91,3.4},{-0.08,6.44,-4.75},{8.55,-0.77,4.66},{-9.89,-8.74,-7.01},{-7.96,0.95,7.43},{0.04,-0.93,2.77},{9.04,-2.37,-4.64},{6.26,-8.58,-9.26},{8.79,1.59,-0.54},{2.47,3.21,-2.82}},
        { {-2.4,4.12,2.49 },{-4.5,-3.22,9.07},{8.23,3.75,6.46},{-1.19,5.46,-7.21},{-7.16,-3.16,-0.67},{-9.4,1.88,6.72 },{2.39,1.27,-10.0},{9.55,1.42,2.3},{-3.57,3.,-5.04},{-5.01,-4.98,0.11}},
        { {9.87,7.55,-8.48},{-3.5,-6.02,-7.57},{3.4,5.82,4.63},{-3.2,-0.57,5.22},{-1.63,7.42,-7.59},{9.4,-2.44,-9.08},{-6.64,-7.42,3.47},{9.35,-2.07,7.19},{8.58,-9.92,-3.92},{2.,0.9,4.47}},
        { {-1.31,8.34,2.35},{-8.78,-9.22,7.58},{7.15,-1.11,-6.23},{8.73,-9.34,4.52},{-0.18,-9.88,8.38},{-4.89,-9.52,-2.42},{6.98,-9.17,2.01},{3.5,9.11,2.65},{-6.53,1.64,-2.79},{2.38,-1.8,-6.29}},
        { {-8.68,8.26,4.94},{7.49,-3.7,6.07},{-7.26,-0.92,1.11},{2.04,-4.78,2.27},{-0.77,8.68,0.12},{3.,0.83,-9.32},{-1.75,8.48,-6.31},{3.34,-7.74,-5.14},{1.82,-0.09,6.36},{9.18,-5.11,-8.6}},
        { {-8.71,-4.18,3.85},{-9.03,-6.25,5.53},{-8.79,5.75,1.31},{-1.76,-1.4,-3.47},{-8.87,-3.7,-5.4},{0.95,9.43,5.59},{-7.39,-7.35,-8.36},{-0.81,5.47,-7.43},{-3.73,-4.33,-6.09},{-6.04,4.23,-8.14}}
    };

    float inputVectorB_h[TEST_CASES][NUMEL_B][N_DIMS] = {
        { {-9.57,-0.59,8.41},{-6.53,6.68,8.02},{5.11,-9.24,-1.19},{-3.61,0.16,9.63},{-2.25,-1.03,5.41},{-6.86,7.33,9.88},{3.78,-8.85,8.71},{2.06,5.51,-6.19},{-8.9,-7.37,-8.82},{5.0,6.25,-2.29} },
        { {-4.2,7.89,4.},{5.15,9.23,-9.98},{-6.59,-7.03,2.68},{4.27,2.22,2.26},{3.8,-8.31,6.78},{3.02,3.67,-9.28},{5.66,8.19,-8.3},{-6.92,5.01,-1.67},{-0.43,-3.39,3.41},{0.65,7.78,-0.58} },
        { {8.49,-9.61,2.52},{6.22,9.2,-5.7},{3.58,-2.73,-8.28},{1.68,-9.99,-7.69},{-5.01,-3.29,6.66},{-1.97,-0.42,8.51},{-8.29,5.32,8.94},{5.01,-7.73,5.46},{8.08,9.91,-4.69},{-7.84,6.52,0.11} },
        { {-3.65,9.13,0.49},{5.2,-9.82,-1.8},{5.15,3.55,-1.13},{9.52,8.68,-7.16},{7.89,8.92,-9.82},{5.88,7.32,-4.42},{0.16,9.05,-7.79},{-9.61,9.82,9.6},{3.9,5.6,6.54},{9.47,-2.26,-9.42} },
        { {5.16,-9.4,7.01},{-6.74,-5.58,6.86},{-7.52,7.99,-4.3},{-2.87,0.84,-8.52},{-8.65,4.42,4.61},{2.04,-6.77,6.32},{0.66,-6.78,8.48},{1.47,-4.78,-6.51},{-5.37,6.01,-0.72},{1.88,-0.22,-6.58}},
        { {9.26,8.7,-4.33},{-8.07,0.91,7.17},{-5.81,-3.63,-7.7},{-8.4,2.07,-8.9},{-0.73,0.83,4.67},{7.6,-4.02,6.84},{2.74,-7.79,-0.22},{-5.51,-8.62,-2.62},{4.26,-6.13,5.19},{8.43,6.77,-0.31}},
        { {-2.45,3.65,3.51},{7.96,3.24,5.56},{1.05,-7.81,0.05},{7.89,-0.54,4.28},{-5.26,5.28,-8.05},{9.05,-5.79,-5.28},{0.76,-8.21,5.34},{-3.54,8.98,7.75},{-3.52,9.01,4.49},{7.9,3.74,1.18}},
        { {-2.56,9.32,5.24},{-1.44,-2.26,9.79},{-3.15,-0.66,-5.08},{-9.61,-3.97,-0.63},{-0.69,1.23,-8.07},{2.89,-3.38,1.94},{1.59,-4.22,1.52},{-5.25,2.93,-4.45},{-7.22,8.42,-1.66},{-7.79,5.01,-3.55}},
        { {-4.58,8.49,-9.35},{-8.96,-4.87,7.35},{4.99,-3.15,1.41},{-8.95,-5.62,2.89},{4.3,-0.67,5.26},{-9.95,-3.37,-5.77},{-4.,-3.29,-7.81},{-5.18,-3.62,-8.81},{-4.6,-4.07,-2.68},{-8.51,0.89,0.34}}
    };

    // Use a CUDA stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int BLOCK_SIZE = 32; // number of threads per block

    //Declaring device variables and allocating device memory for inputs
    float* inputVectorA_d, * inputVectorB_d;
    int* nearestNeighborIndex_h, * nearestNeighborIndex_d;
    CUDA_CHECK(cudaMallocAsync(&inputVectorA_d, numberOfPointsA * N_DIMS * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&inputVectorB_d, numberOfPointsB * N_DIMS * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&nearestNeighborIndex_d, numberOfPointsA * sizeof(float), stream));
    nearestNeighborIndex_h = (int*)malloc(numberOfPointsA * sizeof(int));

    // Loop running through each test
    size_t numBlocks = ((numberOfPointsA / BLOCK_SIZE) + 1);
    for (int i = 0; i < testCaseCount; i++) {

        // Copy data from host to device
        CUDA_CHECK(cudaMemcpyAsync(inputVectorA_d, inputVectorA_h[i], numberOfPointsA * N_DIMS * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(inputVectorB_d, inputVectorB_h[i], numberOfPointsB * N_DIMS * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Calling nearest neighbor kernel
        void* args[] = { &inputVectorA_d, (void*)&numberOfPointsA, &inputVectorB_d, (void*)&numberOfPointsB, &nearestNeighborIndex_d };
        CUDA_CHECK(cudaLaunchKernel((void*)k_nearestNeighbors, numBlocks, BLOCK_SIZE, args, N_DIMS * numberOfPointsB * sizeof(float), stream));
        
        // Copying memory back to host from device
        CUDA_CHECK(cudaMemcpyAsync(nearestNeighborIndex_h, nearestNeighborIndex_d, numberOfPointsA * sizeof(int), cudaMemcpyDeviceToHost, stream));
		
		// Check tasks in the stream has completed
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Verify the test point with manually computed outputs.
        for (int k = 0; k < numberOfPointsA; k++) {
            assert(outputIndices[i][k] == nearestNeighborIndex_h[k]);
        }
    }

    // Free allocated memory
    CUDA_CHECK(cudaFreeAsync(inputVectorA_d, stream));
    CUDA_CHECK(cudaFreeAsync(inputVectorB_d, stream));
    CUDA_CHECK(cudaFreeAsync(nearestNeighborIndex_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    free(nearestNeighborIndex_h);
}

void benchmark() {
    const int nA = 10000;
    const int nB = 4096;
    const int BLOCK_SIZE = 256;
    const int WARMUP = 3;
    const int TIMED = 100;

    float* hostA = (float*)malloc(nA * N_DIMS * sizeof(float));
    float* hostB = (float*)malloc(nB * N_DIMS * sizeof(float));
    srand(42);
    for (int i = 0; i < nA * N_DIMS; i++) hostA[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;
    for (int i = 0; i < nB * N_DIMS; i++) hostB[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;

    float *devA, *devB;
    int *devIdx;
    CUDA_CHECK(cudaMalloc(&devA, nA * N_DIMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, nB * N_DIMS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devIdx, nA * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(devA, hostA, nA * N_DIMS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devB, hostB, nB * N_DIMS * sizeof(float), cudaMemcpyHostToDevice));

    size_t sharedSize = N_DIMS * nB * sizeof(float);
    size_t numBlocks = (nA + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaFuncSetAttribute(
        (void*)k_nearestNeighbors,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedSize));

    void* args[] = { &devA, (void*)&nA, &devB, (void*)&nB, &devIdx };

    for (int i = 0; i < WARMUP; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_nearestNeighbors,
                   numBlocks, BLOCK_SIZE, args, sharedSize, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    nvtxRangePushA("bench_region");
    for (int i = 0; i < TIMED; i++) {
        CUDA_CHECK(cudaLaunchKernel((void*)k_nearestNeighbors,
                   numBlocks, BLOCK_SIZE, args, sharedSize, 0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devIdx));
    free(hostA);
    free(hostB);
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}