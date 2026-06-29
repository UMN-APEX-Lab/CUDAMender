#include "temperature_distribution.h"
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <assert.h>
#include <nvtx3/nvToolsExt.h>

#undef NDEBUG

// Tolerance for floating-point comparison
#define TOLERANCE               (1e-2)
// Number of threads per block
#define BLOCK_SIZE              (16)

#define CUDA_CHECK(call)                                        \
do {                                                            \
        cudaError_t error = call;                               \
        if (error != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",       \
                    __FILE__, __LINE__,                         \
                    cudaGetErrorString(error));                 \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
} while(0)

void launch() {
    // Number of test cases
    const int TEST_CASE_COUNT = 9;
    // Input number of elements of the plate in x direction
    int numPlateElementsX[TEST_CASE_COUNT] = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    int numPlateElementsY[TEST_CASE_COUNT];

    float boundaryTemperatureElementTopLeft[TEST_CASE_COUNT] =  {
        273.0,    // test case 1
        300.0,    // test case 2
        320.0,    // test case 3
        360.0,    // test case 4
        400.0,    // test case 5
        410.0,    // test case 6
        440.0,    // test case 7
        450.0,    // test case 8
        470.0};    // test case 9

    float boundaryTemperatureElementBelowTopLeft[TEST_CASE_COUNT] =  {
        373.0,    // test case 1
        350.0,    // test case 2
        320.0,    // test case 3
        460.0,    // test case 4
        500.0,    // test case 5
        600.0,    // test case 6
        700.0,    // test case 7
        800.0,    // test case 8
        900.0};   // test case 9

    int numIterations[TEST_CASE_COUNT] = {4, 5, 3, 6, 4, 3, 5, 7, 8};

    // Consider a 2D square plate, so numPlateElementsY will be same as numPlateElementsX
    std::memcpy(numPlateElementsY, numPlateElementsX, TEST_CASE_COUNT * sizeof(int));
    int maxNumPlateElementsX = *std::max_element(numPlateElementsX, numPlateElementsX + TEST_CASE_COUNT);
    int maxNumPlateElementsY = maxNumPlateElementsX;
    
    //Number of elements are greater than allocated device memory, consider increasing NUM_DEVICE_MEMORY_ELEM value
    if(maxNumPlateElementsX * maxNumPlateElementsY > NUM_DEVICE_MEMORY_ELEM) {
        assert(false && "Number of elements are greater than allocated device memory, consider increasing NUM_DEVICE_MEMORY_ELEM value");
    }

    // Expected results for each test
    float expectedTemperatureDistribution[TEST_CASE_COUNT][maxNumPlateElementsX * maxNumPlateElementsY] =  {
        {273.0, 306.333, 373.0, 339.667},
        {300.0, 307.143, 314.286, 350.0, 328.571, 321.429, 342.857, 335.714, 328.571},
        {320.0, 320.0, 320.0, 320.0, 320.0, 280.0, 280.0, 320.0, 320.0, 280.0, 280.0, 320.0, 320.0, 320.0, 320.0, 320.0},
        {360.0, 366.667, 373.333, 380.0, 386.667, 460.0, 374.583, 346.927, 355.573, 393.333, 453.333, 376.406, 335.833, 353.594, 400.0, 446.667, 393.594, 369.74, 374.583, 406.667, 440.0, 433.333, 426.667, 420.0, 413.333},
        {400.0, 405.263, 410.526, 415.789, 421.053, 426.316, 500.0, 350.082, 273.335, 269.243, 330.14, 431.579, 494.737, 303.063, 187.418, 180.633, 276.933, 436.842, 489.474, 307.155, 194.202, 187.418, 283.121, 442.105, 484.21, 370.025, 299.465, 293.277, 350.082, 447.368, 478.947, 473.684, 468.421, 463.158, 457.895, 452.632},
        {410.0, 418.261, 426.522, 434.783, 443.043, 451.304, 459.565, 600, 365.937, 242.853, 213.696, 237.69, 330.312, 467.826, 591.739, 298.098, 127.283, 71.0326, 114.891, 250.598, 476.087, 583.478, 279.524, 88.0706, 31.8206, 75.6793, 234.606, 484.348, 575.217, 303.261, 139.674, 83.4239, 127.283, 264.022, 492.609, 566.956, 401.562, 290.353, 258.614, 276.929, 365.937, 500.87, 558.696, 550.435, 542.174, 533.913, 525.652, 517.391, 509.13},
        {440.0, 449.63, 459.259, 468.889, 478.519, 488.148, 497.778, 507.407, 700.0, 462.546, 337.473, 287.512, 286.59, 326.272, 408.304, 517.037, 690.371, 420.34, 239.132, 152.297, 146.599, 211.353, 342.936, 526.667, 680.741, 392.836, 186.754, 88.6921, 81.4511, 156.115, 316.081, 536.296, 671.111, 393.757, 192.452, 95.9332, 88.6921, 163.319, 324.375, 545.926, 661.482, 431.54, 266.911, 182.936, 175.732, 239.132, 370.443, 555.556, 651.852, 516.788, 414.876, 364.267, 355.972, 387.37, 462.546, 565.185, 642.222, 632.593, 622.963, 613.333, 603.704, 594.074, 584.445, 574.815},
        {450.0, 461.29, 472.581, 483.871, 495.161, 506.452, 517.742, 529.032, 540.323, 800.0, 536.418, 402.995, 341.383, 323.29, 340.182, 385.078, 459.613, 551.613, 788.71, 519.57, 330.103, 223.077, 189.062, 208.688, 283.178, 405.597, 562.903, 777.42, 494.961, 277.621, 153.35, 108.613, 134.055, 222.989, 377.967, 574.194, 766.129, 482.877, 264.214, 133.421, 88.6845, 113.685, 210.585, 374.229, 585.484, 754.839, 496.163, 292.01, 172.645, 128.35, 153.35, 242.206, 398.508, 596.774, 743.549, 537.487, 377.029, 277.709, 242.692, 258.491, 330.103, 450.924, 608.065, 732.258, 613.223, 516.968, 458.377, 431.938, 437.836, 471.641, 536.418, 619.355, 720.968, 709.678, 698.387, 687.097, 675.807, 664.516, 653.226, 641.936, 630.645},
        {470.0, 482.286, 494.571, 506.857, 519.143, 531.429, 543.714, 556.0, 568.286, 580.571, 900.0, 598.148, 450.604, 377.099, 350.257, 352.48, 377.998, 428.708, 502.461, 592.857, 887.714, 597.057, 387.671, 266.379, 212.1, 208.003, 245.626, 325.772, 451.425, 605.143, 875.428, 572.863, 337.611, 190.334, 123.72, 116.53, 160.923, 262.755, 420.587, 617.428, 863.143, 556.172, 311.15, 156.719, 84.6422, 76.7293, 125.475, 236.409, 411.804, 629.714, 850.857, 553.949, 315.248, 163.909, 92.5551, 84.6422, 133.571, 245.107, 422.125, 642, 838.571, 571.964, 358.365, 219.745, 154.964, 146.869, 190.334, 293.043, 453.394, 654.286, 826.285, 618.953, 449.57, 341.235, 286.841, 278.143, 310.948, 387.671, 512.438, 666.571, 814.0, 693.835, 596.235, 529.376, 494.625, 484.304, 496.569, 535.223, 598.148, 678.857, 801.714, 789.428, 777.143, 764.857, 752.571, 740.286, 728, 715.714, 703.428, 691.143}
    };

    // Use a CUDA stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Declare host and device pointers
    float *plateCurrentTemperatures_h, *deviceArray_d, *alternateDataArray_d;
    plateCurrentTemperatures_h = (float*) malloc(maxNumPlateElementsX * maxNumPlateElementsY * sizeof(float));
    
    // Get pointers to the global __device__ array
    cudaGetSymbolAddress((void**) &deviceArray_d, plateCurrentTemperatures_d);
    cudaGetSymbolAddress((void**) &alternateDataArray_d, alternateBuffer_d);

    // Loop to execute each test case
    for (int testCaseId = 0; testCaseId < TEST_CASE_COUNT; testCaseId++) {

        // Initialize inner plate temperatures to zero kelvin
        for (int y = 1; y < numPlateElementsY[testCaseId] - 1; y++) {
            memset(&plateCurrentTemperatures_h[y * numPlateElementsX[testCaseId] + 1], 0, (numPlateElementsX[testCaseId] - 2) * sizeof(float));
        }

        float baseGradient = 1.0f;
        int numberOfEdges = 4;
        int boundaryAdjustment = 5;
        float temperatureGradient = baseGradient / (numberOfEdges * numPlateElementsX[testCaseId] - boundaryAdjustment);
        float temperatureChange = (boundaryTemperatureElementBelowTopLeft[testCaseId] - boundaryTemperatureElementTopLeft[testCaseId]) * temperatureGradient;
        float boundaryTemperature = boundaryTemperatureElementTopLeft[testCaseId];

        // Initialize the boundary temperatures by constantly changing the temperature from the
        // top-left corner in clockwise direction along the boundary till the element below top-left element
        for (int j = 0; j < numPlateElementsX[testCaseId] - 1; j++) {
            plateCurrentTemperatures_h[j] = boundaryTemperature;
            boundaryTemperature += temperatureChange;
        }

        for (int j = 0; j < numPlateElementsY[testCaseId] - 1; j++) {
            plateCurrentTemperatures_h[(j + 1) * numPlateElementsX[testCaseId] - 1] = boundaryTemperature;
            boundaryTemperature += temperatureChange;
        }

        for (int j = numPlateElementsX[testCaseId] - 1; j >= 1; j--) {
            plateCurrentTemperatures_h[(numPlateElementsY[testCaseId] - 1) * (numPlateElementsX[testCaseId]) + j] = boundaryTemperature;
            boundaryTemperature += temperatureChange;
        }

        for (int j = numPlateElementsY[testCaseId] - 1; j >= 1; j--) {
            plateCurrentTemperatures_h[j * numPlateElementsX[testCaseId]] = boundaryTemperature;
            boundaryTemperature += temperatureChange;
        }

        // Copying data into device memory
        CUDA_CHECK(cudaMemcpyAsync(deviceArray_d, plateCurrentTemperatures_h, numPlateElementsX[testCaseId] * numPlateElementsY[testCaseId] * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(alternateDataArray_d, plateCurrentTemperatures_h, numPlateElementsX[testCaseId] * numPlateElementsY[testCaseId] * sizeof(float), cudaMemcpyHostToDevice, stream));

        // Determine the number of threads and blocks
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
        dim3 gridSize((numPlateElementsX[testCaseId] + BLOCK_SIZE - 1) / BLOCK_SIZE, (numPlateElementsY[testCaseId] + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Launch the kernel
        // Grid: ((numPlateElementsX[testCaseId] + BLOCK_SIZE - 1) / BLOCK_SIZE, (numPlateElementsY[testCaseId] + BLOCK_SIZE - 1) / BLOCK_SIZE)
        // Block: (BLOCK_SIZE, BLOCK_SIZE, 1)
        void *args[] = {&deviceArray_d, (void*) &numPlateElementsX[testCaseId], (void*) &numPlateElementsY[testCaseId], (void*) &numIterations[testCaseId]};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_temperatureDistribution, gridSize, blockSize, args, 0, stream));

        // Copy the output array plateUpdatedTemperatures_d from the device (GPU) to the host (CPU)
        CUDA_CHECK(cudaMemcpyAsync(plateCurrentTemperatures_h, deviceArray_d, numPlateElementsX[testCaseId] * numPlateElementsY[testCaseId] * sizeof(float), cudaMemcpyDeviceToHost, stream));

        // Check tasks in the stream has completed
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Verify whether the calculated plateCurrentTemperatures_h (computed by GPU) matches the expected result or not
        for (int i = 0; i < numPlateElementsX[testCaseId] * numPlateElementsY[testCaseId]; i++) {
            assert(fabs(plateCurrentTemperatures_h[i] - expectedTemperatureDistribution[testCaseId][i]) < TOLERANCE);
        }
    }

    // Free host memories
    free(plateCurrentTemperatures_h);

    // Free stream
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void benchmark() {
    const int N = 32;
    const int NUM_ITERS = 500;
    const int WARMUP = 3;
    const int TIMED = 100;

    float *plateCurrentTemperatures_h = (float*) malloc(N * N * sizeof(float));

    float *deviceArray_d, *alternateDataArray_d;
    cudaGetSymbolAddress((void**) &deviceArray_d, plateCurrentTemperatures_d);
    cudaGetSymbolAddress((void**) &alternateDataArray_d, alternateBuffer_d);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Initialize boundary temperatures
    auto init_plate = [&]() {
        for (int y = 0; y < N; y++) {
            for (int x = 0; x < N; x++) {
                if (y == 0 || y == N - 1 || x == 0 || x == N - 1) {
                    plateCurrentTemperatures_h[y * N + x] = 400.0f + 2.0f * (x + y);
                } else {
                    plateCurrentTemperatures_h[y * N + x] = 0.0f;
                }
            }
        }
    };

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int numPlateX = N;
    int numPlateY = N;
    int numIter = NUM_ITERS;

    // Warmup
    for (int w = 0; w < WARMUP; w++) {
        init_plate();
        CUDA_CHECK(cudaMemcpyAsync(deviceArray_d, plateCurrentTemperatures_h, N * N * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(alternateDataArray_d, plateCurrentTemperatures_h, N * N * sizeof(float), cudaMemcpyHostToDevice, stream));
        void *args[] = {&deviceArray_d, (void*) &numPlateX, (void*) &numPlateY, (void*) &numIter};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_temperatureDistribution, gridSize, blockSize, args, 0, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Timed region
    nvtxRangePushA("bench_region");
    for (int t = 0; t < TIMED; t++) {
        init_plate();
        CUDA_CHECK(cudaMemcpyAsync(deviceArray_d, plateCurrentTemperatures_h, N * N * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(alternateDataArray_d, plateCurrentTemperatures_h, N * N * sizeof(float), cudaMemcpyHostToDevice, stream));
        void *args[] = {&deviceArray_d, (void*) &numPlateX, (void*) &numPlateY, (void*) &numIter};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)k_temperatureDistribution, gridSize, blockSize, args, 0, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    free(plateCurrentTemperatures_h);
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}