#include "spsv_sell.h"
#include <cstring>
#include <nvtx3/nvToolsExt.h>
#include <cmath>

static void generate_lower_tri_sell(int N, int slice_size,
                                    std::vector<int> &sliceOffsets,
                                    std::vector<int> &columns,
                                    std::vector<float> &values,
                                    int &nnz) {
  int num_slices = (N + slice_size - 1) / slice_size;
  sliceOffsets.resize(num_slices + 1);
  columns.clear();
  values.clear();
  nnz = 0;

  sliceOffsets[0] = 0;
  for (int s = 0; s < num_slices; s++) {
    int row_start = s * slice_size;
    int row_end = std::min(row_start + slice_size, N);
    int max_nnz_in_slice = 0;
    for (int r = row_start; r < row_end; r++) {
      int row_nnz = r + 1;
      if (row_nnz > max_nnz_in_slice) max_nnz_in_slice = row_nnz;
    }
    int slice_values_size = max_nnz_in_slice * slice_size;
    int offset = (int)columns.size();
    columns.resize(offset + slice_values_size, -1);
    values.resize(offset + slice_values_size, 0.0f);
    for (int c = 0; c < max_nnz_in_slice; c++) {
      for (int lr = 0; lr < slice_size; lr++) {
        int r = row_start + lr;
        int idx = offset + c * slice_size + lr;
        if (r < N && c <= r) {
          columns[idx] = c;
          values[idx] = (c == r) ? 1.0f : 0.01f;
          nnz++;
        }
      }
    }
    sliceOffsets[s + 1] = (int)columns.size();
  }
}

int main(int argc, char *argv[]) {

bool perf_mode = false;
for (int i = 1; i < argc; i++) {
  if (std::strcmp(argv[i], "--perf") == 0) perf_mode = true;
}

if (perf_mode) {
  const int N = 512;
  const int slice_size = 2;
  int nnz = 0;
  std::vector<int> sliceOffsets, cols;
  std::vector<float> vals;
  generate_lower_tri_sell(N, slice_size, sliceOffsets, cols, vals, nnz);

  std::vector<float> hX(N);
  for (int i = 0; i < N; i++) hX[i] = 1.0f;
  float alpha = 1.0f;

  const int warmup = 3;
  const int timed = 10;

  for (int w = 0; w < warmup; w++) {
    std::vector<float> hY(N, 0.0f);
    spsv_sell_example(N, N, nnz, sliceOffsets, cols, vals, hX, hY, alpha);
  }

  nvtxRangePushA("bench_region");
  for (int t = 0; t < timed; t++) {
    std::vector<float> hY(N, 0.0f);
    spsv_sell_example(N, N, nnz, sliceOffsets, cols, vals, hX, hY, alpha);
  }
  cudaDeviceSynchronize();
  nvtxRangePop();

  return 0;
}
auto spsv_sell_test = []() {
  const int A_num_rows = 4;
  const int A_num_cols = 4;
  const int A_nnz = 9;
  const int A_slice_size = 2;
  const int A_values_size = 12;
  const int A_num_slices = (A_num_rows + A_slice_size - 1) / A_slice_size; // 2

  std::vector<int> hA_sliceOffsets = {0, 6, 12};
  std::vector<int> hA_columns = {
      // Slice 0
      0, 1, 2, -1, 3, -1,
      // Slice 1
      0, 1, 2, 3, 3, -1};
  std::vector<float> hA_values = {1.0f, 4.0f, 2.0f, 0.0f, 3.0f, 0.0f,
                                  5.0f, 8.0f, 6.0f, 9.0f, 7.0f, 0.0f};
  std::vector<float> hX = {1.0f, 8.0f, 23.0f, 52.0f};
  std::vector<float> hY_result = {1.0f, 2.0f, 3.0f, 4.0f};

  std::vector<float> hY(A_num_rows, 0.0f);

  float alpha = 1.0f;

  spsv_sell_example(A_num_rows, A_num_cols, A_nnz, hA_sliceOffsets, hA_columns,
                    hA_values, hX, hY, alpha);

  bool correct = true;
  for (int i = 0; i < A_num_rows; i++) {
    if (hY[i] != hY_result[i]) {
      correct = false;
      break;
    }
  }

  if (correct) {
    std::cout << "spsv_sell_example test PASSED" << std::endl;
  } else {
    std::cout << "spsv_sell_example test FAILED: wrong result" << std::endl;
    std::exit(EXIT_FAILURE);
  }
};

spsv_sell_test();

}