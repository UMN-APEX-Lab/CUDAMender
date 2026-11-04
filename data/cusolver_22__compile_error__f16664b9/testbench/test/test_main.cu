#include "compute_svd.h"
#include <cstring>

static void run_benchmark() {
  const int m = 256, n = 128;
  const int ldu = m, ldv = n;
  const int minmn = (m < n) ? m : n;

  std::vector<double> A_orig(m * n);
  srand(42);
  for (int i = 0; i < m * n; ++i) {
    A_orig[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  std::vector<double> A(m * n);
  std::vector<double> S(minmn);
  std::vector<double> U(ldu * m);
  std::vector<double> V(ldv * n);
  double h_err_sigma;

  const int warmup = 3;
  const int timed = 100;

  for (int i = 0; i < warmup; ++i) {
    A = A_orig;
    S.assign(minmn, 0.0);
    U.assign(ldu * m, 0.0);
    V.assign(ldv * n, 0.0);
    compute_svd(m, n, A, S, U, V, h_err_sigma);
  }

  for (int i = 0; i < timed; ++i) {
    A = A_orig;
    S.assign(minmn, 0.0);
    U.assign(ldu * m, 0.0);
    V.assign(ldv * n, 0.0);
    compute_svd(m, n, A, S, U, V, h_err_sigma);
  }
}

int main(int argc, char *argv[]) {
  if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }
auto test_compute_svd = []() {
  const int m = 3, n = 2;
  const int ldu = m, ldv = n;
  const int minmn = (m < n) ? m : n;

  std::vector<double> A = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
  std::vector<double> S(n, 0);
  std::vector<double> U(ldu * m, 0);
  std::vector<double> V(ldv * n, 0);
  double h_err_sigma;

  compute_svd(m, n, A, S, U, V, h_err_sigma);

  printf("Singular values S:\n");
  for (int i = 0; i < n; ++i) {
    printf("%f ", S[i]);
  }
  printf("\n");

  printf("Left singular vectors U:\n");
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < minmn; ++j) {
      printf("%f ", U[i * ldu + j]);
    }
    printf("\n");
  }

  printf("Right singular vectors V:\n");
  for (int i = 0; i < minmn; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%f ", V[i * ldv + j]);
    }
    printf("\n");
  }

  printf("h_err_sigma: %E\n", h_err_sigma);
};

test_compute_svd();

}