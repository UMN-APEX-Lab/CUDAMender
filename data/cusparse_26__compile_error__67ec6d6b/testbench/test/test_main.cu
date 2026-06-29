#include "spsv_csr.h"
#include "cusparse_utils.h"
#include <vector>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

static void generate_lower_triangular_csr(int N,
                                          std::vector<int>& csrOffsets,
                                          std::vector<int>& columns,
                                          std::vector<float>& values) {
  csrOffsets.resize(N + 1);
  columns.clear();
  values.clear();
  int nnz = 0;
  for (int i = 0; i < N; i++) {
    csrOffsets[i] = nnz;
    for (int j = 0; j <= i; j++) {
      columns.push_back(j);
      values.push_back((i == j) ? static_cast<float>(i + 1) : 0.5f);
      nnz++;
    }
  }
  csrOffsets[N] = nnz;
}

static void compute_rhs(int N,
                         const std::vector<int>& csrOffsets,
                         const std::vector<int>& columns,
                         const std::vector<float>& values,
                         const std::vector<float>& y_true,
                         std::vector<float>& x) {
  x.resize(N);
  for (int i = 0; i < N; i++) {
    float sum = 0.0f;
    for (int idx = csrOffsets[i]; idx < csrOffsets[i + 1]; idx++) {
      sum += values[idx] * y_true[columns[idx]];
    }
    x[i] = sum;
  }
}

int main(int argc, char* argv[]) {

bool perf_mode = false;
for (int i = 1; i < argc; i++) {
  if (std::strcmp(argv[i], "--perf") == 0) {
    perf_mode = true;
  }
}

if (perf_mode) {
  const int N = 500;
  std::vector<int> csrOffsets, columns;
  std::vector<float> values;
  generate_lower_triangular_csr(N, csrOffsets, columns, values);
  int nnz = csrOffsets[N];

  std::vector<float> y_true(N);
  for (int i = 0; i < N; i++) y_true[i] = static_cast<float>(i + 1);
  std::vector<float> hX;
  compute_rhs(N, csrOffsets, columns, values, y_true, hX);

  const int warmup_iters = 3;
  const int timed_iters = 100;

  for (int it = 0; it < warmup_iters; it++) {
    std::vector<float> hY(N, 0.0f);
    spsv_csr_example(N, N, nnz, csrOffsets, columns, values, hX, hY);
  }

  nvtxRangePushA("bench_region");
  for (int it = 0; it < timed_iters; it++) {
    std::vector<float> hY(N, 0.0f);
    spsv_csr_example(N, N, nnz, csrOffsets, columns, values, hX, hY);
  }
  nvtxRangePop();

  return 0;
}

auto spsv_csr_test = []() {
  const int A_num_rows = 4;
  const int A_num_cols = 4;
  const int A_nnz = 9;

  std::vector<int> hA_csrOffsets = {0, 3, 4, 7, 9};
  std::vector<int> hA_columns = {0, 2, 3, 1, 0, 2, 3, 1, 3};
  std::vector<float> hA_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                  6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<float> hX = {1.0f, 8.0f, 23.0f, 52.0f};
  std::vector<float> hY(A_num_rows, 0.0f);
  std::vector<float> hY_result = {1.0f, 2.0f, 3.0f, 4.0f};

  spsv_csr_example(A_num_rows, A_num_cols, A_nnz, hA_csrOffsets, hA_columns,
                   hA_values, hX, hY);

  bool correct = true;
  for (int i = 0; i < A_num_rows; i++) {
    if (hY[i] != hY_result[i]) {
      correct = false;
      break;
    }
  }
  if (correct)
    std::cout << "spsv_csr_example test PASSED" << std::endl;
  else
    std::cout << "spsv_csr_example test FAILED: wrong result" << std::endl;
};

spsv_csr_test();

}