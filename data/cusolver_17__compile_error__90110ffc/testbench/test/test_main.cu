#include "compute_eigenvalues.h"
#include "cusolver_helpers.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <nvtx3/nvToolsExt.h>

static void run_correctness_test() {
    const int64_t n = 3;
    const std::vector<double> A = {1.0, 7.0, 4.0, 2.0, 4.0, 2.0, -3.0, -2.0, 1.0};
    std::vector<double> W(2 * n, 0);  // 2*n for real and imaginary parts
    std::vector<double> VR(n * n, 0); // n*n for eigenvectors

    compute_eigenvalues_and_vectors(n, A, W, VR);

    printf("Eigenvalues (real + imaginary parts):\n");
    for (int64_t i = 0; i < n; ++i) {
      printf("W[%ld] = %f + %fi\n", i, W[i], W[n + i]);
    }

    printf("Eigenvectors:\n");
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t j = 0; j < n; ++j) {
        printf("%f ", VR[i + j * n]);
      }
      printf("\n");
    }
}

static void run_benchmark() {
    const int64_t n = 256;
    const int warmup_iters = 3;
    const int timed_iters = 10;

    srand(42);
    std::vector<double> A(n * n);
    for (int64_t i = 0; i < n * n; ++i) {
        A[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    std::vector<double> W(2 * n, 0);
    std::vector<double> VR(n * n, 0);

    for (int i = 0; i < warmup_iters; ++i) {
        std::fill(W.begin(), W.end(), 0.0);
        std::fill(VR.begin(), VR.end(), 0.0);
        compute_eigenvalues_and_vectors(n, A, W, VR);
    }
    cudaDeviceSynchronize();

    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed_iters; ++i) {
        std::fill(W.begin(), W.end(), 0.0);
        std::fill(VR.begin(), VR.end(), 0.0);
        compute_eigenvalues_and_vectors(n, A, W, VR);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
}

int main(int argc, char *argv[]) {
    bool perf = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--perf") == 0) {
            perf = true;
        }
    }

    if (perf) {
        run_benchmark();
    } else {
        run_correctness_test();
    }

    return 0;
}