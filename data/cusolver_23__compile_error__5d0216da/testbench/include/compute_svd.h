#ifndef COMPUTE_SVD_H
#define COMPUTE_SVD_H

#include <vector>
#include <cstdint>

void compute_svd(int64_t m, int64_t n, int64_t rank, int64_t p, int64_t iters,
                 std::vector<double> &A, std::vector<double> &S,
                 std::vector<double> &U, std::vector<double> &V);

#endif // COMPUTE_SVD_H