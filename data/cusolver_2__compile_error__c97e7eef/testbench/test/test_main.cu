#include "compute_svd_batched.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

static void verify_svd_reconstruction(int batchSize, int m, int n,
                                      const std::vector<float> &A,
                                      const std::vector<float> &S,
                                      const std::vector<float> &U,
                                      const std::vector<float> &V) {
  const float atol = 1e-3f;
  const float rtol = 1e-3f;
  for (int b = 0; b < batchSize; b++) {
    const float *Ab = A.data() + b * m * n;
    const float *Sb = S.data() + b * n;
    const float *Ub = U.data() + b * m * n;
    const float *Vb = V.data() + b * n * n;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        float reconstructed = 0.0f;
        for (int k = 0; k < n; k++) {
          reconstructed += Ub[i + k * m] * Sb[k] * Vb[j + k * n];
        }
        float expected = Ab[i + j * m];
        float abs_err = fabsf(reconstructed - expected);
        float rel_err = abs_err / (fabsf(expected) + 1e-8f);
        assert((abs_err <= atol || rel_err <= rtol) &&
               "SVD reconstruction exceeds tolerance");
      }
    }
  }
}

static void test_svd_batched() {
  const int batchSize = 2;
  const int m = 3;
  const int n = 2;

  std::vector<float> A = {1.0,  4.0, 2.0, 2.0, 5.0, 1.0,
                          10.0, 8.0, 6.0, 9.0, 7.0, 5.0};
  std::vector<float> S(n * batchSize, 0);
  std::vector<float> U(m * n * batchSize, 0);
  std::vector<float> V(n * n * batchSize, 0);

  compute_svd_batched(batchSize, m, n, A, S, U, V);

  std::printf("Singular values for first matrix:\n");
  for (int i = 0; i < n; i++) {
    std::printf("%f ", S[i]);
  }
  std::printf("\n");

  std::printf("Singular values for second matrix:\n");
  for (int i = 0; i < n; i++) {
    std::printf("%f ", S[n + i]);
  }
  std::printf("\n");

  for (int i = 0; i < n * batchSize; i++) {
    assert(S[i] > 0.0f && "Singular values must be positive");
  }

  verify_svd_reconstruction(batchSize, m, n, A, S, U, V);

  std::printf("Correctness test PASSED.\n");
}

static void run_benchmark() {
  const int batchSize = 256;
  const int m = 32;
  const int n = 32;
  const int total = m * n * batchSize;

  std::vector<float> A(total);
  srand(42);
  for (int i = 0; i < total; i++) {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  std::vector<float> S(n * batchSize, 0);
  std::vector<float> U(m * n * batchSize, 0);
  std::vector<float> V(n * n * batchSize, 0);

  const int warmup = 3;
  const int timed = 10;

  for (int i = 0; i < warmup; i++) {
    compute_svd_batched(batchSize, m, n, A, S, U, V);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed; i++) {
    compute_svd_batched(batchSize, m, n, A, S, U, V);
  }
  nvtxRangePop();
}

int main(int argc, char **argv) {
  if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
  } else {
    test_svd_batched();
  }
  return 0;
}
