#include "compute_svd.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

static void run_benchmark() {
  const int64_t m = 512, n = 512, rank = 32, p = 8, iters = 2;
  const int warmup = 3;
  const int timed = 100;

  std::vector<double> A_template(m * n);
  srand(42);
  for (auto &v : A_template) v = (double)rand() / RAND_MAX;

  for (int i = 0; i < warmup; i++) {
    std::vector<double> A(A_template);
    std::vector<double> S(std::min(m, n), 0);
    std::vector<double> U(m * rank, 0);
    std::vector<double> V(n * rank, 0);
    compute_svd(m, n, rank, p, iters, A, S, U, V);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed; i++) {
    std::vector<double> A(A_template);
    std::vector<double> S(std::min(m, n), 0);
    std::vector<double> U(m * rank, 0);
    std::vector<double> V(n * rank, 0);
    compute_svd(m, n, rank, p, iters, A, S, U, V);
  }
  nvtxRangePop();
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }

auto test_compute_svd = []() {
  const int64_t m = 5, n = 5, rank = 2, p = 2, iters = 2;
  std::vector<double> A = {
      0.76420743, 0.61411544, 0.81724151, 0.42040879, 0.03446089,
      0.03697287, 0.85962444, 0.67584086, 0.45594666, 0.02074835,
      0.42018265, 0.39204509, 0.12657948, 0.90250559, 0.23076218,
      0.50339844, 0.92974961, 0.21213988, 0.63962457, 0.58124562,
      0.58325673, 0.11589871, 0.39831112, 0.21492685, 0.00540355};
  std::vector<double> S(std::min(m, n), 0);
  std::vector<double> U(m * m, 0);
  std::vector<double> V(n * n, 0);

  compute_svd(m, n, rank, p, iters, A, S, U, V);

  printf("Singular values:\n");
  for (const auto &val : S) {
    printf("%f ", val);
  }
  printf("\n");
};

test_compute_svd();

}