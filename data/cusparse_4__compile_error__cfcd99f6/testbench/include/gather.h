#ifndef GATHER_H
#define GATHER_H

#include <vector>

void gather(int size, int nnz, const std::vector<int>& hX_indices,
            const std::vector<float>& hY, std::vector<float>& hX_values);

#endif // GATHER_H