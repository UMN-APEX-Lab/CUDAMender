#include "conjugate_gradient.h"
#include <vector>
#include <cmath>
#include <cstring>
#undef NDEBUG
#include <assert.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Structure to define a test case
struct TestCase {
    int matrixSize;
    std::vector<double> matrixA;
    std::vector<double> vectorB;
    std::vector<double> expectedVectorX;
};

void launch() {
    // Initialize cuBLAS
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    // Define all test cases
    std::vector<TestCase> testCases;

    // Test Case 1: Simple 2x2 Matrix
    testCases.push_back({
        2,
        {4.0, 1.0,
         1.0, 3.0},
        {1.0, 2.0},
        {0.09091, 0.63636}
    });

    // Test Case 2: Diagonal Matrix
    testCases.push_back({
        3,
        {10.0, 0.0, 0.0,
         0.0, 20.0, 0.0,
         0.0, 0.0, 30.0},
        {10.0, 40.0, 90.0},
        {1.0, 2.0, 3.0}
    });

    // Test Case 3: Symmetric Positive-Definite Matrix
    testCases.push_back({
        3,
        {6.0, 2.0, 1.0,
         2.0, 5.0, 2.0,
         1.0, 2.0, 4.0},
        {14.0, 18.0, 17.0},
        {1.19277, 1.92771, 2.98795}
    });

    // Test Case 4: Matrix with Negative Entries
    testCases.push_back({
        2,
        {4.0, -1.0,
         -1.0, 3.0},
        {1.0, 2.0},
        {0.45455, 0.81818}
    });

    // Test Case 5: Identity Matrix
    testCases.push_back({
        4,
        {1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0},
        {5.0, 10.0, 15.0, 20.0},
        {5.0, 10.0, 15.0, 20.0}
    });

    // Test Case 6: All Ones Matrix
    testCases.push_back({
        3,
        {2.0, 1.0, 1.0,
         1.0, 2.0, 1.0,
         1.0, 1.0, 2.0},
        {6.0, 6.0, 6.0},
        {1.5, 1.5, 1.5}
    });

    // Test Case 7: Larger Symmetric Matrix
    testCases.push_back({
        5,
        {6.0, 2.0, 1.0, 0.0, 0.0,
         2.0, 5.0, 2.0, 0.0, 0.0,
         1.0, 2.0, 4.0, 1.0, 0.0,
         0.0, 0.0, 1.0, 3.0, 1.0,
         0.0, 0.0, 0.0, 1.0, 2.0},
        {14.0, 18.0, 17.0, 12.0, 8.0},
        {1.22039, 2.20386, 2.26995, 2.29202, 2.85399}
    });

    // Test Case 8: Near-Zero Diagonal Matrix
    testCases.push_back({
        3,
        {1e-6, 0.0, 0.0,
         0.0, 1e-6, 0.0,
         0.0, 0.0, 1e-6},
        {1e-6, 2e-6, 3e-6},
        {1.0, 2.0, 3.0}
    });

    // Test Case 9: Matrix with Repeated Eigenvalues
    testCases.push_back({
        2,
        {5.0, 4.0,
         4.0, 5.0},
        {9.0, 9.0},
        {1.0, 1.0}
    });

    // Test Case 10: Symmetric Positive-Definite Matrix
    testCases.push_back({
        2,
        {2.0, 1.0,
         1.0, 3.0},
        {1.0, 2.0},
        {0.2, 0.6}
    });

    // Create a CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    size_t maxMatrixSize = 5;

    // Allocate device memory
    double *matrixA_d, *vectorX_d, *residual_d, *searchDir_d, *productAdir_d;
    CUDA_CHECK(cudaMallocAsync((void **)&matrixA_d, maxMatrixSize * maxMatrixSize * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync((void **)&vectorX_d, maxMatrixSize * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync((void **)&residual_d, maxMatrixSize * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync((void **)&searchDir_d, maxMatrixSize * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync((void **)&productAdir_d, maxMatrixSize * sizeof(double), stream));

    // Run all test cases
    for (const auto &currTestCase : testCases) {
        int matrixSize = currTestCase.matrixSize;

        // Initialize vectors
        std::vector<double> vectorX(matrixSize, 0.0);
        std::vector<double> residualVector(matrixSize, 0.0);
        std::vector<double> searchDirection(matrixSize, 0.0);
        std::vector<double> productAdir(matrixSize, 0.0);

        // Convert matrixA from row-major to column-major
        std::vector<double> columnMajorA(matrixSize * matrixSize, 0.0);
        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                columnMajorA[j * matrixSize + i] = currTestCase.matrixA[i * matrixSize + j];
            }
        }

        // Copy matrix matrixA and vector vectorB to device
        CUDA_CHECK(cudaMemcpyAsync(matrixA_d, columnMajorA.data(), matrixSize * matrixSize * sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(residual_d, currTestCase.vectorB.data(), matrixSize * sizeof(double), cudaMemcpyHostToDevice, stream));

        // Initialize vectorX to zeros on both host and device
        std::fill(vectorX.begin(), vectorX.end(), 0.0);
        CUDA_CHECK(cudaMemcpyAsync(vectorX_d, vectorX.data(), matrixSize * sizeof(double), cudaMemcpyHostToDevice, stream));

        // residualVector = vectorB - matrixA * vectorX = vectorB (since vectorX is initialized to zero)
        // searchDirection = residualVector
        CUDA_CHECK(cudaMemcpyAsync(searchDir_d, residual_d, matrixSize * sizeof(double), cudaMemcpyDeviceToDevice, stream));

        // Bind the cuBLAS handle to stream
        CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));

        k_conjugateGradientKernel(cublasHandle, matrixSize, residual_d, matrixA_d, searchDir_d, vectorX_d, productAdir_d);

        // Copy solution vectorX back to host
        CUDA_CHECK(cudaMemcpyAsync(vectorX.data(), vectorX_d, matrixSize * sizeof(double), cudaMemcpyDeviceToHost, stream));

        // Compute CPU solution for verification using CPU Conjugate Gradient
        std::vector<double> vectorXCpu(matrixSize, 0.0);
        std::vector<double> vectorRCpu = currTestCase.vectorB;
        std::vector<double> searchDirCpu = vectorRCpu;
        std::vector<double> productADirCpu(matrixSize, 0.0);

        double oldResidualCpu = 0.0;
        for (int i = 0; i < matrixSize; i++) {
            oldResidualCpu += vectorRCpu[i] * vectorRCpu[i];
        }

        double newResidualCpu, aphaCpu, betaCpu;

        int iteratorCpu = 0;
        while (iteratorCpu < MAX_ITERATIONS) {
            // productADirCpu = matrixA * searchDirCpu
            for (int i = 0; i < matrixSize; i++) {
                productADirCpu[i] = 0.0;
                for (int j = 0; j < matrixSize; j++) {
                    productADirCpu[i] += currTestCase.matrixA[i * matrixSize + j] * searchDirCpu[j];
                }
            }

            // Compute aphaCpu = oldResidualCpu / (searchDirCpu^T * productADirCpu)
            double pAp_cpu = 0.0;
            for (int i = 0; i < matrixSize; i++) {
                pAp_cpu += searchDirCpu[i] * productADirCpu[i];
            }

            if (pAp_cpu == 0.0) {
                break;
            }
            aphaCpu = oldResidualCpu / pAp_cpu;

            // Update vectorXCpu = vectorXCpu + aphaCpu * searchDirCpu
            for (int i = 0; i < matrixSize; i++) {
                vectorXCpu[i] += aphaCpu * searchDirCpu[i];
            }

            // Update vectorRCpu = vectorRCpu - aphaCpu * productADirCpu
            for (int i = 0; i < matrixSize; i++) {
                vectorRCpu[i] -= aphaCpu * productADirCpu[i];
            }

            // Compute newResidualCpu = vectorRCpu^T * vectorRCpu
            newResidualCpu = 0.0;
            for (int i = 0; i < matrixSize; i++) {
                newResidualCpu += vectorRCpu[i] * vectorRCpu[i];
            }

            // Check for convergence
            if (std::sqrt(newResidualCpu) < TOLERANCE) {
                break;
            }

            // Compute betaCpu = newResidualCpu / oldResidualCpu
            betaCpu = newResidualCpu / oldResidualCpu;

            // Update searchDirCpu = vectorRCpu + betaCpu * searchDirCpu
            for (int i = 0; i < matrixSize; i++) {
                searchDirCpu[i] = vectorRCpu[i] + betaCpu * searchDirCpu[i];
            }

            // Update oldResidualCpu for next iteration
            oldResidualCpu = newResidualCpu;
            iteratorCpu++;
        }

        // Compare GPU solution with expected solution
        bool solutionMatched = true;
        for (int i = 0; i < matrixSize; i++) {
            if (std::fabs(vectorX[i] - currTestCase.expectedVectorX[i]) > TOLERANCE) {
                solutionMatched = false;
            }
        }

        // Compare GPU solution with CPU solution
        bool cpuGpuMatched = true;
        for (int i = 0; i < matrixSize; i++) {
            if (std::fabs(vectorX[i] - vectorXCpu[i]) > TOLERANCE) {
                cpuGpuMatched = false;
            }
        }

        // Handle the case where the solution is close but not exact due to floating-point precision
        if (!solutionMatched) {
            // Calculate relative error
            bool relativeErrorOk = true;
            for (int i = 0; i < matrixSize; i++) {
                double expected = currTestCase.expectedVectorX[i];
                double computed = vectorX[i];
                double relativeError = (expected != 0.0) ? std::fabs((computed - expected) / expected) : std::fabs(computed);
                if (relativeError > TOLERANCE) {
                    relativeErrorOk = false;
                }
            }
            assert(relativeErrorOk == true);
        }

        // Additionally, compare CPU and GPU solutions
        if (cpuGpuMatched == false) {
            assert(cpuGpuMatched);
        }
    }

    // Free device memory
    CUDA_CHECK(cudaFreeAsync(matrixA_d, stream));
    CUDA_CHECK(cudaFreeAsync(vectorX_d, stream));
    CUDA_CHECK(cudaFreeAsync(residual_d, stream));
    CUDA_CHECK(cudaFreeAsync(searchDir_d, stream));
    CUDA_CHECK(cudaFreeAsync(productAdir_d, stream));

    // Destroy the CUDA stream
    CUDA_CHECK(cudaStreamDestroy(stream));

    // Synchronize the device to ensure all operations are complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Destroy cuBLAS cublasHandle
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
}

void generateSPDMatrix(std::vector<double>& A, int n, unsigned seed) {
    srand(seed);
    std::vector<double> R(n * n);
    for (int i = 0; i < n * n; i++)
        R[i] = ((double)rand() / RAND_MAX) - 0.5;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
                sum += R[k * n + i] * R[k * n + j];
            A[i * n + j] = sum;
        }
        A[i * n + i] += (double)n;
    }
}

void benchmark() {
    const int matrixSize = 512;
    const int warmup = 3;
    const int timed = 100;

    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));

    std::vector<double> matrixA(matrixSize * matrixSize);
    std::vector<double> vectorB(matrixSize);
    generateSPDMatrix(matrixA, matrixSize, 42);
    for (int i = 0; i < matrixSize; i++) {
        vectorB[i] = 1.0;
    }

    std::vector<double> columnMajorA(matrixSize * matrixSize);
    for (int i = 0; i < matrixSize; i++)
        for (int j = 0; j < matrixSize; j++)
            columnMajorA[j * matrixSize + i] = matrixA[i * matrixSize + j];

    double *matrixA_d, *vectorX_d, *residual_d, *searchDir_d, *productAdir_d;
    cudaMalloc(&matrixA_d, matrixSize * matrixSize * sizeof(double));
    cudaMalloc(&vectorX_d, matrixSize * sizeof(double));
    cudaMalloc(&residual_d, matrixSize * sizeof(double));
    cudaMalloc(&searchDir_d, matrixSize * sizeof(double));
    cudaMalloc(&productAdir_d, matrixSize * sizeof(double));

    cudaMemcpy(matrixA_d, columnMajorA.data(), matrixSize * matrixSize * sizeof(double), cudaMemcpyHostToDevice);

    auto runOnce = [&]() {
        cudaMemcpy(residual_d, vectorB.data(), matrixSize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(vectorX_d, 0, matrixSize * sizeof(double));
        cudaMemcpy(searchDir_d, residual_d, matrixSize * sizeof(double), cudaMemcpyDeviceToDevice);
        k_conjugateGradientKernel(cublasHandle, matrixSize, residual_d, matrixA_d, searchDir_d, vectorX_d, productAdir_d);
        cudaDeviceSynchronize();
    };

    for (int i = 0; i < warmup; i++)
        runOnce();

    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed; i++)
        runOnce();
    nvtxRangePop();

    cudaFree(matrixA_d);
    cudaFree(vectorX_d);
    cudaFree(residual_d);
    cudaFree(searchDir_d);
    cudaFree(productAdir_d);
    cudaStreamDestroy(stream);
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}