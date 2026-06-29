#ifndef SDDMM_BSR_H
#define SDDMM_BSR_H

#include <vector>

void sddmm_bsr_example(int A_num_rows, int A_num_cols, int B_num_rows,
                       int B_num_cols, int row_block_dim, int col_block_dim,
                       const std::vector<float> &hA,
                       const std::vector<float> &hB,
                       const std::vector<int> &hC_boffsets,
                       const std::vector<int> &hC_bcolumns,
                       std::vector<float> &hC_values);

#endif // SDDMM_BSR_H