#include "eigen_decomposition.h"
#include <cstring>
#include <nvtx3/nvToolsExt.h>
#include <random>

static void generate_symmetric_matrix(int m, std::vector<double> &A) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  A.resize(static_cast<size_t>(m) * m);
  for (int i = 0; i < m; ++i) {
    for (int j = i; j < m; ++j) {
      double val = dist(rng);
      A[i * m + j] = val;
      A[j * m + i] = val;
    }
  }
}

static void run_benchmark() {
  const int m = 512;
  std::vector<double> A;
  generate_symmetric_matrix(m, A);

  const int warmup = 3;
  const int timed = 100;

  std::vector<double> W(m);
  std::vector<double> V(static_cast<size_t>(m) * m);

  for (int i = 0; i < warmup; ++i) {
    compute_eigen_decomposition(m, A, W, V);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed; ++i) {
    compute_eigen_decomposition(m, A, W, V);
  }
  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }

auto test_compute_eigen_decomposition = []() {
  const int m = 3;
  const int lda = m;
  std::vector<double> A = {3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0};
  std::vector<double> W(m, 0);
  std::vector<double> V(lda * m, 0);

  compute_eigen_decomposition(m, A, W, V);

  std::printf("Eigenvalues:\n");
  for (int i = 0; i < m; ++i) {
    std::printf("W[%d] = %E\n", i, W[i]);
  }

  std::printf("Eigenvectors:\n");
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      std::printf("%E ", V[i * m + j]);
    }
    std::printf("\n");
  }
};

test_compute_eigen_decomposition();

}