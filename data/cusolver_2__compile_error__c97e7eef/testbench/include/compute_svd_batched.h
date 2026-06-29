#ifndef COMPUTE_SVD_BATCHED_H
#define COMPUTE_SVD_BATCHED_H

#include <vector>
#include "cusolver_utils.h"

void compute_svd_batched(int batchSize, int m, int n,
                         const std::vector<float> &A, std::vector<float> &S,
                         std::vector<float> &U, std::vector<float> &V);

#endif // COMPUTE_SVD_BATCHED_H