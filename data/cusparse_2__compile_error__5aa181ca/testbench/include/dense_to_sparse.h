#ifndef DENSE_TO_SPARSE_H
#define DENSE_TO_SPARSE_H

#include <vector>

void dense_to_sparse_blocked_ell(int num_rows, int num_cols, int ell_blk_size,
                                 int ell_width,
                                 const std::vector<float> &h_dense,
                                 const std::vector<int> &h_ell_columns,
                                 std::vector<float> &h_ell_values);

#endif // DENSE_TO_SPARSE_H