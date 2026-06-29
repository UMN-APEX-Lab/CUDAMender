#include "sddmm_bsr.h"
#include <cstring>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <vector>

static void run_benchmark() {
  const int A_num_rows = 512;
  const int A_num_cols = 512;
  const int B_num_rows = A_num_cols;
  const int B_num_cols = 512;
  const int row_block_dim = 16;
  const int col_block_dim = 16;
  const int C_num_brows = A_num_rows / row_block_dim;
  const int C_num_bcols = B_num_cols / col_block_dim;

  std::vector<int> hC_boffsets(C_num_brows + 1);
  std::vector<int> hC_bcolumns;
  for (int i = 0; i < C_num_brows; i++) {
    hC_boffsets[i] = static_cast<int>(hC_bcolumns.size());
    hC_bcolumns.push_back(i % C_num_bcols);
    hC_bcolumns.push_back((i + 1) % C_num_bcols);
  }
  hC_boffsets[C_num_brows] = static_cast<int>(hC_bcolumns.size());
  const int C_bnnz = static_cast<int>(hC_bcolumns.size());
  const int C_nnz = C_bnnz * row_block_dim * col_block_dim;

  const int lda = A_num_rows;
  const int ldb = B_num_cols;
  const int A_size = lda * A_num_cols;
  const int B_size = ldb * B_num_rows;

  std::vector<float> hA(A_size);
  std::vector<float> hB(B_size);
  for (int i = 0; i < A_size; i++) hA[i] = static_cast<float>((i % 17) + 1);
  for (int i = 0; i < B_size; i++) hB[i] = static_cast<float>((i % 13) + 1);

  std::vector<float> hC_values(C_nnz, 0.0f);

  const int warmup_iters = 3;
  const int timed_iters = 100;

  for (int i = 0; i < warmup_iters; i++) {
    sddmm_bsr_example(A_num_rows, A_num_cols, B_num_rows, B_num_cols,
                       row_block_dim, col_block_dim, hA, hB, hC_boffsets,
                       hC_bcolumns, hC_values);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed_iters; i++) {
    sddmm_bsr_example(A_num_rows, A_num_cols, B_num_rows, B_num_cols,
                       row_block_dim, col_block_dim, hA, hB, hC_boffsets,
                       hC_bcolumns, hC_values);
  }
  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }
auto test_sddmm_bsr_example = []() {
  const int A_num_rows = 8;
  const int A_num_cols = 8;
  const int B_num_rows = A_num_cols;
  const int B_num_cols = 8;
  const int row_block_dim = 4;
  const int col_block_dim = 4;
  const int C_num_brows = A_num_rows / row_block_dim;
  const int C_num_bcols = B_num_cols / col_block_dim;
  const int C_bnnz = 2;
  const int C_nnz = C_bnnz * row_block_dim * col_block_dim;
  const int lda = A_num_rows;
  const int ldb = B_num_cols;
  const int A_size = lda * A_num_cols;
  const int B_size = ldb * B_num_rows;

  const std::vector<float> hA = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f,
      2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
      8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> hB = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f,
      2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f,
      5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
      8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<int> hC_boffsets = {0, 1, 2};
  const std::vector<int> hC_bcolumns = {0, 1};

  std::vector<float> hC_values(C_nnz, 0.0f);

  sddmm_bsr_example(A_num_rows, A_num_cols, B_num_rows, B_num_cols,
                    row_block_dim, col_block_dim, hA, hB, hC_boffsets,
                    hC_bcolumns, hC_values);

  std::cout << "SDDMM BSR test completed successfully." << std::endl;
  for (int i = 0; i < C_nnz; i++) {
    std::cout << "C[" << i << "] = " << hC_values[i] << std::endl;
  }
};

test_sddmm_bsr_example();

}