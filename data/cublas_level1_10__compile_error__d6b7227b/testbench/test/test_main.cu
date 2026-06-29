#include "cublas_rotm.h"
#include "cuda_helpers.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <nvtx3/nvToolsExt.h>

int main(int argc, char *argv[]) {
  bool perf_mode = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--perf") == 0) {
      perf_mode = true;
    }
  }

  if (perf_mode) {
    const int n = 2000000;
    const std::vector<double> param = {1.0, 5.0, 6.0, 7.0, 8.0};

    const int warmup_iters = 3;
    const int timed_iters = 100;

    for (int i = 0; i < warmup_iters; ++i) {
      std::vector<double> A(n, 1.0);
      std::vector<double> B(n, 2.0);
      cublas_rotm_example(n, A, B, param);
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed_iters; ++i) {
      std::vector<double> A(n, 1.0);
      std::vector<double> B(n, 2.0);
      cublas_rotm_example(n, A, B, param);
    }
    nvtxRangePop();

    return 0;
  }

  auto test_cublas_rotm_example = []() {
    const int n = 4;
    std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> B = {5.0, 6.0, 7.0, 8.0};
    const std::vector<double> param = {1.0, 5.0, 6.0, 7.0, 8.0};

    cublas_rotm_example(n, A, B, param);

    printf("A after rotation:\n");
    for (const auto &val : A) {
      printf("%f ", val);
    }
    printf("\n");

    printf("B after rotation:\n");
    for (const auto &val : B) {
      printf("%f ", val);
    }
    printf("\n");
  };

  test_cublas_rotm_example();

  return 0;
}