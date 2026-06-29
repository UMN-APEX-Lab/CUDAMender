#ifndef SPSV_CSR_H
#define SPSV_CSR_H

#include <vector>

void spsv_csr_example(int A_num_rows, int A_num_cols, int A_nnz,
                      const std::vector<int> &hA_csrOffsets,
                      const std::vector<int> &hA_columns,
                      const std::vector<float> &hA_values,
                      const std::vector<float> &hX, std::vector<float> &hY);

#endif // SPSV_CSR_H