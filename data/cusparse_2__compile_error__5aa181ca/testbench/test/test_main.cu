#include "dense_to_sparse.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

static void run_correctness_test() {
    const int num_rows = 4;
    const int num_cols = 6;
    const int ell_blk_size = 2;
    const int ell_width = 4;
    const int ld = num_cols;
    const int dense_size = ld * num_rows;
    const int nnz = ell_width * num_rows;

    std::vector<float> h_dense = {0.0f, 0.0f,  1.0f, 2.0f, 0.0f,  0.0f,
                                   0.0f, 0.0f,  3.0f, 4.0f, 0.0f,  0.0f,
                                   5.0f, 6.0f,  0.0f, 0.0f, 7.0f,  8.0f,
                                   9.0f, 10.0f, 0.0f, 0.0f, 11.0f, 12.0f};
    std::vector<int> h_ell_columns = {1, -1, 0, 2};
    std::vector<float> h_ell_values(nnz, 0.0f);
    std::vector<float> h_ell_values_result = {
        1.0f, 2.0f, 0.0f, 0.0f, 3.0f, 4.0f,  0.0f,  0.0f,
        5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    dense_to_sparse_blocked_ell(num_rows, num_cols, ell_blk_size, ell_width,
                                h_dense, h_ell_columns, h_ell_values);

    bool correct = true;
    for (int i = 0; i < nnz; i++) {
      if (h_ell_values[i] != h_ell_values_result[i]) {
        correct = false;
        break;
      }
    }

    if (correct) {
      std::cout << "Test PASSED" << std::endl;
    } else {
      std::cout << "Test FAILED" << std::endl;
      std::exit(EXIT_FAILURE);
    }
}

static void run_benchmark() {
    const int num_rows = 512;
    const int num_cols = 512;
    const int ell_blk_size = 2;
    const int num_block_rows = num_rows / ell_blk_size;
    const int num_block_cols = num_cols / ell_blk_size;
    const int ell_cols_per_block_row = num_block_cols;
    const int ell_width = ell_cols_per_block_row * ell_blk_size;
    const int nnz = ell_width * num_rows;

    std::vector<float> h_dense(num_rows * num_cols);
    for (int i = 0; i < num_rows * num_cols; i++) {
        h_dense[i] = static_cast<float>((i % 17) + 1);
    }

    std::vector<int> h_ell_columns(num_block_rows * ell_cols_per_block_row);
    for (int br = 0; br < num_block_rows; br++) {
        for (int j = 0; j < ell_cols_per_block_row; j++) {
            h_ell_columns[br * ell_cols_per_block_row + j] = j;
        }
    }

    std::vector<float> h_ell_values(nnz, 0.0f);

    const int warmup_iters = 3;
    const int timed_iters = 100;

    for (int i = 0; i < warmup_iters; i++) {
        std::fill(h_ell_values.begin(), h_ell_values.end(), 0.0f);
        dense_to_sparse_blocked_ell(num_rows, num_cols, ell_blk_size,
                                    ell_width, h_dense, h_ell_columns,
                                    h_ell_values);
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed_iters; i++) {
        std::fill(h_ell_values.begin(), h_ell_values.end(), 0.0f);
        dense_to_sparse_blocked_ell(num_rows, num_cols, ell_blk_size,
                                    ell_width, h_dense, h_ell_columns,
                                    h_ell_values);
    }
    nvtxRangePop();
}

int main(int argc, char *argv[]) {
    bool perf_mode = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--perf") == 0) {
            perf_mode = true;
        }
    }

    if (perf_mode) {
        run_benchmark();
    } else {
        run_correctness_test();
    }

    return 0;
}