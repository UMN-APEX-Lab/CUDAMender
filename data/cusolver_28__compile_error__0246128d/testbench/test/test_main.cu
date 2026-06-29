#include "trtri.h"
#include "cuda_helpers.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <nvtx3/nvToolsExt.h>

static void run_correctness_test() {
  const int n = 4;
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  const cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

  std::vector<double> A = {4.0, 1.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0,
                           0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  trtri(n, A, uplo, diag);

  printf("Inverted matrix:\n");
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", A[i * n + j]);
    }
    printf("\n");
  }
}

static void run_benchmark() {
  const int n = 1024;
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;
  const cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;

  const int total = n * n;
  std::vector<double> A_template(total, 0.0);
  srand(42);
  for (int i = 0; i < n; ++i) {
    A_template[i * n + i] = 1.0 + (rand() % 100) / 10.0;
    for (int j = i + 1; j < n; ++j) {
      A_template[i * n + j] = (rand() % 100) / 100.0;
    }
  }

  const int warmup = 3;
  const int timed = 100;

  for (int i = 0; i < warmup; ++i) {
    std::vector<double> A_copy(A_template);
    trtri(n, A_copy, uplo, diag);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed; ++i) {
    std::vector<double> A_copy(A_template);
    trtri(n, A_copy, uplo, diag);
  }
  nvtxRangePop();
}

int main(int argc, char* argv[]) {
  bool perf = false;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--perf") == 0) perf = true;
  }

  if (perf) {
    run_benchmark();
  } else {
    run_correctness_test();
  }

  return 0;
}