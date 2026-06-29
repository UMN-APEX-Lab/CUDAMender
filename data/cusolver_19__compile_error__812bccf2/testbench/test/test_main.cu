#include "cusolver_eigen.h"
#include <cstring>
#include <nvtx3/nvToolsExt.h>

static void generate_complex_matrix(int64_t n, std::vector<cuDoubleComplex>& A) {
  A.resize(n * n);
  for (int64_t j = 0; j < n; ++j) {
    for (int64_t i = 0; i < n; ++i) {
      double re = static_cast<double>((i * 31 + j * 17 + 7) % 1000) / 500.0 - 1.0;
      double im = static_cast<double>((i * 13 + j * 23 + 3) % 1000) / 500.0 - 1.0;
      A[i + j * n] = make_cuDoubleComplex(re, im);
    }
  }
}

static void run_benchmark() {
  const int64_t n = 256;
  std::vector<cuDoubleComplex> A;
  generate_complex_matrix(n, A);

  const int warmup = 3;
  const int timed = 7;

  std::vector<cuDoubleComplex> W(n);
  std::vector<cuDoubleComplex> VR(n * n);

  for (int i = 0; i < warmup; ++i) {
    compute_eigenvalues_and_vectors(n, A, W, VR);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed; ++i) {
    compute_eigenvalues_and_vectors(n, A, W, VR);
  }
  nvtxRangePop();
}

int main(int argc, char* argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }
auto test_compute_eigenvalues_and_vectors = []() {
  const int64_t n = 3;
  std::vector<cuDoubleComplex> A = {{2.0, 1.0},  {2.0, 1.0},  {1.0, 2.0},
                                    {-1.0, 0.0}, {-3.0, 1.0}, {-1.0, 2.0},
                                    {1.0, 2.0},  {2.0, 3.0},  {0.0, 1.0}};
  std::vector<cuDoubleComplex> W(n);
  std::vector<cuDoubleComplex> VR(n * n);

  compute_eigenvalues_and_vectors(n, A, W, VR);

  printf("Eigenvalues:\n");
  for (int i = 0; i < n; ++i) {
    printf("(%f, %f)\n", cuCreal(W[i]), cuCimag(W[i]));
  }

  printf("Right Eigenvectors:\n");
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("(%f, %f) ", cuCreal(VR[i + j * n]), cuCimag(VR[i + j * n]));
    }
    printf("\n");
  }
};

test_compute_eigenvalues_and_vectors();

}