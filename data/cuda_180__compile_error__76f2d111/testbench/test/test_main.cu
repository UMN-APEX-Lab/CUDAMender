#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <random>
#include <cstring>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <nvtx3/nvToolsExt.h>
#include "graph_config.h"
#include "graph_ops.h"

constexpr float ERROR_TOLERANCE = 1e-3;
constexpr int DETERMINISTIC_INITIAL_CONDITION = 42;

void launch() {

    int deviceId = 0;
    cudaDeviceProp deviceProperties;
    int maxActiveCooperativeBlocksPerSM = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(   &maxActiveCooperativeBlocksPerSM, 
                                                                k_calculate, 
                                                                BLOCK_SIZE, 
                                                                0));
    int maxActiveBlocks = maxActiveCooperativeBlocksPerSM * deviceProperties.multiProcessorCount;
    
    // The stream to be utilized by CUDA-graph.
    cudaStream_t stream;
    // Allocating the stream.
    CUDA_CHECK(cudaStreamCreate(&stream));
    // Allocating the device memory.
    float * dataIn_d;
    float * dataOut_d;
    int * constantParams_d;
    CUDA_CHECK(cudaMallocAsync(&dataIn_d, sizeof(float) * MAX_MATRIX_ELEMENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&dataOut_d, sizeof(float) * MAX_MATRIX_ELEMENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&constantParams_d, sizeof(int), stream));
    // Initializing the buffers to zero.
    CUDA_CHECK(cudaMemsetAsync(dataIn_d, 0, sizeof(float) * MAX_MATRIX_ELEMENTS, stream));
    CUDA_CHECK(cudaMemsetAsync(dataOut_d, 0, sizeof(float) * MAX_MATRIX_ELEMENTS, stream));
    CUDA_CHECK(cudaMemsetAsync(constantParams_d, 0, sizeof(int), stream));

    // Allocating the host memory.
    float * data_h = new float[MAX_MATRIX_ELEMENTS];
    int * constantParams_h = new int[1];
    float * expectedData_h = new float[MAX_MATRIX_ELEMENTS * 2];

    //  Test 1: [ 1 ] matrix to the power of 8 (2 ^ 3) -> [ 2187 ] matrix
    {
        // Initializing the host buffer for input.
        std::initializer_list<float> input = { 
            1.00f, 1.00f, 1.00f, 
            1.00f, 1.00f, 1.00f, 
            1.00f, 1.00f, 1.00f
        };
        std::copy(input.begin(), input.end(), data_h);
        // Setting matrix size to 3x3.
        constantParams_h[SELECT_PARAM_MATRIX_SIZE] = 3;
        int elementsUsed = constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE];
        // Copying inputs from host.
        CUDA_CHECK(cudaMemcpyAsync(dataIn_d, data_h, sizeof(float) * elementsUsed, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(constantParams_d, constantParams_h, sizeof(int), cudaMemcpyHostToDevice, stream));
        // Executing graph to calculate matrix ^ (2 ^ 3).
        runGraph(3, dataIn_d, dataOut_d, constantParams_d, maxActiveBlocks, stream);
        // Copying outputs to host.
        CUDA_CHECK(cudaMemcpyAsync(data_h, dataOut_d, sizeof(float) * elementsUsed, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Comparing results.
        float expectedResult = 2187.0f;
        for(int i = 0; i < elementsUsed; i++) {
            assert(fabsf(expectedResult - data_h[i]) < ERROR_TOLERANCE);
        }
    }
    //  Test 2: Identity matrix to the power of 32 (2 ^ 5) -> Identity matrix.
    {
        // Initializing the host buffer for input.
        std::initializer_list<float> input = { 
            1.00f, 0.00f, 0.00f, 0.00f, 
            0.00f, 1.00f, 0.00f, 0.00f, 
            0.00f, 0.00f, 1.00f, 0.00f, 
            0.00f, 0.00f, 0.00f, 1.00f
        };
        std::copy(input.begin(), input.end(), data_h);
        // Setting matrix size to 4x4.
        constantParams_h[SELECT_PARAM_MATRIX_SIZE] = 4;
        int elementsUsed = constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE];
        // Copying inputs from host.
        CUDA_CHECK(cudaMemcpyAsync(dataIn_d, data_h, sizeof(float) * elementsUsed, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(constantParams_d, constantParams_h, sizeof(int), cudaMemcpyHostToDevice, stream));
        // Executing graph to calculate matrix ^ (2 ^ 5).
        runGraph(5, dataIn_d, dataOut_d, constantParams_d, maxActiveBlocks, stream);
        // Copying outputs to host.
        CUDA_CHECK(cudaMemcpyAsync(data_h, dataOut_d, sizeof(float) * elementsUsed, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Comparing results.
        std::initializer_list<float> expected = { 
            1.00f, 0.00f, 0.00f, 0.00f, 
            0.00f, 1.00f, 0.00f, 0.00f, 
            0.00f, 0.00f, 1.00f, 0.00f, 
            0.00f, 0.00f, 0.00f, 1.00f
        };
        for(int i = 0; i < elementsUsed; i++) {
            assert(fabsf(expected.begin()[i] - data_h[i]) < ERROR_TOLERANCE);
        }
    }
    //  Test 3: Randomized matrix elements, in range of (0.0f, 1.0f).
    {
        // Initializing the host buffer for input.
        std::mt19937 generator(DETERMINISTIC_INITIAL_CONDITION);
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        // Setting matrix size to the maximum allowed.
        constantParams_h[SELECT_PARAM_MATRIX_SIZE] = MAX_MATRIX_SIZE;
        int elementsUsed = constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE];
        for(int i = 0; i < elementsUsed; i++) {
            data_h[i] = distribution(generator);
            expectedData_h[i] = data_h[i];
        }
        // Copying inputs from host.
        CUDA_CHECK(cudaMemcpyAsync(dataIn_d, data_h, sizeof(float) * elementsUsed, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(constantParams_d, constantParams_h, sizeof(int), cudaMemcpyHostToDevice, stream));
        // Executing graph to calculate matrix ^ (2 ^ 2).
        runGraph(2, dataIn_d, dataOut_d, constantParams_d, maxActiveBlocks, stream);
        // Copying outputs to host.
        CUDA_CHECK(cudaMemcpyAsync(data_h, dataOut_d, sizeof(float) * elementsUsed, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Comparing results.
        int pingPongBuffering = 0;
        for(int i = 0; i < 2; i++) {
            int readOffset = (pingPongBuffering % 2) ? MAX_MATRIX_ELEMENTS : 0;
            int writeOffset = (pingPongBuffering % 2) ? 0 : MAX_MATRIX_ELEMENTS;
            pingPongBuffering++;
            for(int e = 0; e < elementsUsed; e++) {
                int outputColumn = e % constantParams_h[SELECT_PARAM_MATRIX_SIZE];
                int outputRow = e / constantParams_h[SELECT_PARAM_MATRIX_SIZE];
                float c = 0.0f;
                for(int i = 0; i < constantParams_h[SELECT_PARAM_MATRIX_SIZE]; i++) {
                    float a = expectedData_h[readOffset + outputRow * constantParams_h[SELECT_PARAM_MATRIX_SIZE] + i];
                    float b = expectedData_h[readOffset + outputColumn + i * constantParams_h[SELECT_PARAM_MATRIX_SIZE]];
                    c = fmaf(a, b, c);
                }
                expectedData_h[writeOffset + outputColumn + outputRow * constantParams_h[SELECT_PARAM_MATRIX_SIZE]] = c;
            }
        }
        for (int i = 0; i < constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE]; i++) {
            float expected = expectedData_h[i + ((pingPongBuffering % 2) ? MAX_MATRIX_ELEMENTS : 0)];
            if(fabsf(expected) > ERROR_TOLERANCE) {
                assert(fabsf(expected - data_h[i]) / fabsf(expected) < ERROR_TOLERANCE);
            } else {
                assert(fabsf(expected - data_h[i]) < ERROR_TOLERANCE);
            }
        }
    }
    //  Test 4: Randomized matrix elements, in range of (-1.0f, 1.0f).
    {
        // Initializing the host buffer for input.
        std::mt19937 generator(DETERMINISTIC_INITIAL_CONDITION);
        std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
        constantParams_h[SELECT_PARAM_MATRIX_SIZE] = MAX_MATRIX_SIZE;
        int elementsUsed = constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE];
        for(int i = 0; i < elementsUsed; i++) {
            data_h[i] = distribution(generator);
            expectedData_h[i] = data_h[i];
        }
        // Copying inputs from host.
        CUDA_CHECK(cudaMemcpyAsync(dataIn_d, data_h, sizeof(float) * elementsUsed, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(constantParams_d, constantParams_h, sizeof(int), cudaMemcpyHostToDevice, stream));
        // Executing graph to calculate matrix ^ (2 ^ 2).
        runGraph(2, dataIn_d, dataOut_d, constantParams_d, maxActiveBlocks, stream);
        // Copying outputs to host.
        CUDA_CHECK(cudaMemcpyAsync(data_h, dataOut_d, sizeof(float) * elementsUsed, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Comparing results.
        int pingPongBuffering = 0;
        for(int i = 0; i < 2; i++) {
            int readOffset = (pingPongBuffering % 2) ? MAX_MATRIX_ELEMENTS : 0;
            int writeOffset = (pingPongBuffering % 2) ? 0 : MAX_MATRIX_ELEMENTS;
            pingPongBuffering++;
            for(int e = 0; e < elementsUsed; e++) {
                int outputColumn = e % constantParams_h[SELECT_PARAM_MATRIX_SIZE];
                int outputRow = e / constantParams_h[SELECT_PARAM_MATRIX_SIZE];
                float c = 0.0f;
                for(int i = 0; i < constantParams_h[SELECT_PARAM_MATRIX_SIZE]; i++) {
                    float a = expectedData_h[readOffset + outputRow * constantParams_h[SELECT_PARAM_MATRIX_SIZE] + i];
                    float b = expectedData_h[readOffset + outputColumn + i * constantParams_h[SELECT_PARAM_MATRIX_SIZE]];
                    c = fmaf(a, b, c);
                }
                expectedData_h[writeOffset + outputColumn + outputRow * constantParams_h[SELECT_PARAM_MATRIX_SIZE]] = c;
            }
        }
        for (int i = 0; i < constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE]; i++) {
            float expected = expectedData_h[i + ((pingPongBuffering % 2) ? MAX_MATRIX_ELEMENTS : 0)];
            if(fabsf(expected) > ERROR_TOLERANCE) {
                assert(fabsf(expected - data_h[i]) / fabsf(expected) < ERROR_TOLERANCE);
            } else {
                assert(fabsf(expected - data_h[i]) < ERROR_TOLERANCE);
            }
        }
    }
    //  Test 5: Randomized matrix elements, in range of (-1000.0f, 1000.0f).
    {
        // Initializing the host buffer for input.
        std::mt19937 generator(DETERMINISTIC_INITIAL_CONDITION);
        std::uniform_real_distribution<float> distribution(-1000.0f, 1000.0f);
        constantParams_h[SELECT_PARAM_MATRIX_SIZE] = MAX_MATRIX_SIZE;
        int elementsUsed = constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE];
        for(int i = 0; i < elementsUsed; i++) {
            data_h[i] = distribution(generator);
            expectedData_h[i] = data_h[i];
        }
        // Copying inputs from host.
        CUDA_CHECK(cudaMemcpyAsync(dataIn_d, data_h, sizeof(float) * elementsUsed, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(constantParams_d, constantParams_h, sizeof(int), cudaMemcpyHostToDevice, stream));
        // Executing graph to calculate matrix ^ (2 ^ 2).
        runGraph(2, dataIn_d, dataOut_d, constantParams_d, maxActiveBlocks, stream);
        // Copying outputs to host.
        CUDA_CHECK(cudaMemcpyAsync(data_h, dataOut_d, sizeof(float) * elementsUsed, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Comparing results.
        int pingPongBuffering = 0;
        for(int i = 0; i < 2; i++) {
            int readOffset = (pingPongBuffering % 2) ? MAX_MATRIX_ELEMENTS : 0;
            int writeOffset = (pingPongBuffering % 2) ? 0 : MAX_MATRIX_ELEMENTS;
            pingPongBuffering++;
            for(int e = 0; e < elementsUsed; e++) {
                int outputColumn = e % constantParams_h[SELECT_PARAM_MATRIX_SIZE];
                int outputRow = e / constantParams_h[SELECT_PARAM_MATRIX_SIZE];
                float c = 0.0f;
                for(int i = 0; i < constantParams_h[SELECT_PARAM_MATRIX_SIZE]; i++) {
                    float a = expectedData_h[readOffset + outputRow * constantParams_h[SELECT_PARAM_MATRIX_SIZE] + i];
                    float b = expectedData_h[readOffset + outputColumn + i * constantParams_h[SELECT_PARAM_MATRIX_SIZE]];
                    c = fmaf(a, b, c);
                }
                expectedData_h[writeOffset + outputColumn + outputRow * constantParams_h[SELECT_PARAM_MATRIX_SIZE]] = c;
            }
        }
        for (int i = 0; i < constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE]; i++) {
            float expected = expectedData_h[i + ((pingPongBuffering % 2) ? MAX_MATRIX_ELEMENTS : 0)];
            if(fabsf(expected) > ERROR_TOLERANCE) {
                assert(fabsf(expected - data_h[i]) / fabsf(expected) < ERROR_TOLERANCE);
            } else {
                assert(fabsf(expected - data_h[i]) < ERROR_TOLERANCE);
            }
        }
    }
    //  Test 6: Randomized matrix elements, in range of (-1000000.0f, 1000000.0f).
    {
        // Initializing the host buffer for input.
        std::mt19937 generator(DETERMINISTIC_INITIAL_CONDITION);
        std::uniform_real_distribution<float> distribution(-1000000.0f, 1000000.0f);
        constantParams_h[SELECT_PARAM_MATRIX_SIZE] = MAX_MATRIX_SIZE;
        int elementsUsed = constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE];
        for(int i = 0; i < elementsUsed; i++) {
            data_h[i] = distribution(generator);
            expectedData_h[i] = data_h[i];
        }
        // Copying inputs from host.
        CUDA_CHECK(cudaMemcpyAsync(dataIn_d, data_h, sizeof(float) * elementsUsed, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(constantParams_d, constantParams_h, sizeof(int), cudaMemcpyHostToDevice, stream));
        // Executing graph to calculate matrix ^ (2 ^ 2).
        runGraph(2, dataIn_d, dataOut_d, constantParams_d, maxActiveBlocks, stream);
        // Copying outputs to host.
        CUDA_CHECK(cudaMemcpyAsync(data_h, dataOut_d, sizeof(float) * elementsUsed, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Comparing results.
        int pingPongBuffering = 0;
        for(int i = 0; i < 2; i++) {
            int readOffset = (pingPongBuffering % 2) ? MAX_MATRIX_ELEMENTS : 0;
            int writeOffset = (pingPongBuffering % 2) ? 0 : MAX_MATRIX_ELEMENTS;
            pingPongBuffering++;
            for(int e = 0; e < elementsUsed; e++) {
                int outputColumn = e % constantParams_h[SELECT_PARAM_MATRIX_SIZE];
                int outputRow = e / constantParams_h[SELECT_PARAM_MATRIX_SIZE];
                float c = 0.0f;
                for(int i = 0; i < constantParams_h[SELECT_PARAM_MATRIX_SIZE]; i++) {
                    float a = expectedData_h[readOffset + outputRow * constantParams_h[SELECT_PARAM_MATRIX_SIZE] + i];
                    float b = expectedData_h[readOffset + outputColumn + i * constantParams_h[SELECT_PARAM_MATRIX_SIZE]];
                    c = fmaf(a, b, c);
                }
                expectedData_h[writeOffset + outputColumn + outputRow * constantParams_h[SELECT_PARAM_MATRIX_SIZE]] = c;
            }
        }
        for (int i = 0; i < constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE]; i++) {
            float expected = expectedData_h[i + ((pingPongBuffering % 2) ? MAX_MATRIX_ELEMENTS : 0)];
            if(fabsf(expected) > ERROR_TOLERANCE) {
                assert(fabsf(expected - data_h[i]) / fabsf(expected) < ERROR_TOLERANCE);
            } else {
                assert(fabsf(expected - data_h[i]) < ERROR_TOLERANCE);
            }
        }
    }
    //  Test 7: Elements with increasing values.
    {
        // Initializing the host buffer for input.
        constantParams_h[SELECT_PARAM_MATRIX_SIZE] = MAX_MATRIX_SIZE;
        int elementsUsed = constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE];
        for(int i = 0; i < elementsUsed; i++) {
            data_h[i] = i;
            expectedData_h[i] = data_h[i];
        }
        // Copying inputs from host.
        CUDA_CHECK(cudaMemcpyAsync(dataIn_d, data_h, sizeof(float) * elementsUsed, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(constantParams_d, constantParams_h, sizeof(int), cudaMemcpyHostToDevice, stream));
        // Executing graph to calculate matrix ^ (2 ^ 2).
        runGraph(2, dataIn_d, dataOut_d, constantParams_d, maxActiveBlocks, stream);
        // Copying outputs to host.
        CUDA_CHECK(cudaMemcpyAsync(data_h, dataOut_d, sizeof(float) * elementsUsed, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Comparing results.
        int pingPongBuffering = 0;
        for(int i = 0; i < 2; i++) {
            int readOffset = (pingPongBuffering % 2) ? MAX_MATRIX_ELEMENTS : 0;
            int writeOffset = (pingPongBuffering % 2) ? 0 : MAX_MATRIX_ELEMENTS;
            pingPongBuffering++;
            for(int e = 0; e < elementsUsed; e++) {
                int outputColumn = e % constantParams_h[SELECT_PARAM_MATRIX_SIZE];
                int outputRow = e / constantParams_h[SELECT_PARAM_MATRIX_SIZE];
                float c = 0.0f;
                for(int i = 0; i < constantParams_h[SELECT_PARAM_MATRIX_SIZE]; i++) {
                    float a = expectedData_h[readOffset + outputRow * constantParams_h[SELECT_PARAM_MATRIX_SIZE] + i];
                    float b = expectedData_h[readOffset + outputColumn + i * constantParams_h[SELECT_PARAM_MATRIX_SIZE]];
                    c = fmaf(a, b, c);
                }
                expectedData_h[writeOffset + outputColumn + outputRow * constantParams_h[SELECT_PARAM_MATRIX_SIZE]] = c;
            }
        }
        for (int i = 0; i < constantParams_h[SELECT_PARAM_MATRIX_SIZE] * constantParams_h[SELECT_PARAM_MATRIX_SIZE]; i++) {
            float expected = expectedData_h[i + ((pingPongBuffering % 2) ? MAX_MATRIX_ELEMENTS : 0)];
            if(fabsf(expected) > ERROR_TOLERANCE) {
                assert(fabsf(expected - data_h[i]) / fabsf(expected) < ERROR_TOLERANCE);
            } else {
                assert(fabsf(expected - data_h[i]) < ERROR_TOLERANCE);
            }
        }
    }


    // Deallocating the resources.
    CUDA_CHECK(cudaFreeAsync(dataIn_d, stream));
    CUDA_CHECK(cudaFreeAsync(dataOut_d, stream));
    CUDA_CHECK(cudaFreeAsync(constantParams_d, stream));
    // Deallocating the host memory.
    delete [] data_h;
    delete [] constantParams_h;
    delete [] expectedData_h;
    // Deallocating the stream.
    CUDA_CHECK(cudaStreamDestroy(stream));

}

void benchmark() {
    constexpr int WARMUP_ITERS = 3;
    constexpr int TIMED_ITERS = 100;
    constexpr int GRAPH_LAUNCHES_PER_ITER = 4;

    int deviceId = 0;
    cudaDeviceProp deviceProperties;
    int maxActiveCooperativeBlocksPerSM = 0;
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceId));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(   &maxActiveCooperativeBlocksPerSM,
                                                                k_calculate,
                                                                BLOCK_SIZE,
                                                                0));
    int maxActiveBlocks = maxActiveCooperativeBlocksPerSM * deviceProperties.multiProcessorCount;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float * dataIn_d;
    float * dataOut_d;
    int * constantParams_d;
    CUDA_CHECK(cudaMallocAsync(&dataIn_d, sizeof(float) * MAX_MATRIX_ELEMENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&dataOut_d, sizeof(float) * MAX_MATRIX_ELEMENTS, stream));
    CUDA_CHECK(cudaMallocAsync(&constantParams_d, sizeof(int), stream));

    float * data_h = new float[MAX_MATRIX_ELEMENTS];
    int * constantParams_h = new int[1];

    std::mt19937 generator(123);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    constantParams_h[SELECT_PARAM_MATRIX_SIZE] = MAX_MATRIX_SIZE;
    int elementsUsed = MAX_MATRIX_SIZE * MAX_MATRIX_SIZE;
    for(int i = 0; i < elementsUsed; i++) {
        data_h[i] = distribution(generator);
    }
    CUDA_CHECK(cudaMemcpyAsync(constantParams_d, constantParams_h, sizeof(int), cudaMemcpyHostToDevice, stream));

    for(int w = 0; w < WARMUP_ITERS; w++) {
        CUDA_CHECK(cudaMemcpyAsync(dataIn_d, data_h, sizeof(float) * elementsUsed, cudaMemcpyHostToDevice, stream));
        runGraph(GRAPH_LAUNCHES_PER_ITER, dataIn_d, dataOut_d, constantParams_d, maxActiveBlocks, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    nvtxRangePushA("bench_region");
    for(int t = 0; t < TIMED_ITERS; t++) {
        CUDA_CHECK(cudaMemcpyAsync(dataIn_d, data_h, sizeof(float) * elementsUsed, cudaMemcpyHostToDevice, stream));
        runGraph(GRAPH_LAUNCHES_PER_ITER, dataIn_d, dataOut_d, constantParams_d, maxActiveBlocks, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(dataIn_d, stream));
    CUDA_CHECK(cudaFreeAsync(dataOut_d, stream));
    CUDA_CHECK(cudaFreeAsync(constantParams_d, stream));
    delete [] data_h;
    delete [] constantParams_h;
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}