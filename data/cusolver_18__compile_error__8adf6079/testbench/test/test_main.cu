#include "compute_eigenvalues.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

static void run_benchmark() {
  const int64_t n = 256;
  const int warmup_iters = 3;
  const int timed_iters = 100;

  std::vector<double> A(n * n);
  std::srand(42);
  for (int64_t i = 0; i < n * n; ++i) {
    A[i] = static_cast<double>(std::rand()) / RAND_MAX - 0.5;
  }

  std::vector<cuDoubleComplex> W(n);
  std::vector<double> VR(n * n);

  for (int i = 0; i < warmup_iters; ++i) {
    compute_eigenvalues_and_vectors(n, A, W, VR);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed_iters; ++i) {
    compute_eigenvalues_and_vectors(n, A, W, VR);
  }
  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }

auto test_compute_eigenvalues_and_vectors = []() {
  const int64_t n = 3;
  const std::vector<double> A = {1.0, 7.0,  4.0,  2.0, 4.0,
                                 2.0, -3.0, -2.0, 1.0}; // column-major order
  std::vector<cuDoubleComplex> W(n);
  std::vector<double> VR(n * n);

  compute_eigenvalues_and_vectors(n, A, W, VR);

  std::printf("Eigenvalues:\n");
  for (int i = 0; i < n; ++i) {
    std::printf("(%f, %f)\n", cuCreal(W[i]), cuCimag(W[i]));
  }

  std::printf("Right Eigenvectors:\n");
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::printf("%f ", VR[i + j * n]);
    }
    std::printf("\n");
  }
};

test_compute_eigenvalues_and_vectors();

}