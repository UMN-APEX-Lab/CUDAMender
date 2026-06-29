#ifndef CUBLAS_ROTM_H
#define CUBLAS_ROTM_H

#include <vector>

void cublas_rotm_example(int n, std::vector<double> &A, std::vector<double> &B,
                         const std::vector<double> &param);

#endif // CUBLAS_ROTM_H