#include "solve_matrix.h"
#include <cstring>
#include <nvtx3/nvToolsExt.h>

static void run_correctness_test() {
  const int N = 1024;
  const int nrhs = 1;

  std::vector<double> hA(N * N);
  std::vector<double> hB(N * nrhs);
  std::vector<double> hX(N * nrhs);

  for (int i = 0; i < N * N; ++i) {
    hA[i] = static_cast<double>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < N * nrhs; ++i) {
    hB[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  solve_matrix(N, nrhs, hA, hB, hX);

  std::cout << "Solution (first 10 elements): ";
  for (int i = 0; i < std::min(10, N); ++i) {
    std::cout << hX[i] << " ";
  }
  std::cout << std::endl;
}

static void run_benchmark() {
  const int N = 2048;
  const int nrhs = 4;
  const int warmup_iters = 3;
  const int timed_iters = 100;

  std::vector<double> hA(N * N);
  std::vector<double> hB(N * nrhs);
  std::vector<double> hX(N * nrhs);

  srand(42);
  for (int i = 0; i < N * N; ++i) {
    hA[i] = static_cast<double>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < N; ++i) {
    hA[i * N + i] += static_cast<double>(N);
  }
  for (int i = 0; i < N * nrhs; ++i) {
    hB[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  for (int i = 0; i < warmup_iters; ++i) {
    solve_matrix(N, nrhs, hA, hB, hX);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed_iters; ++i) {
    solve_matrix(N, nrhs, hA, hB, hX);
  }
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