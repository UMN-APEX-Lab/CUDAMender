#ifndef COMPUTE_EIGENVALUES_H
#define COMPUTE_EIGENVALUES_H

#include <cstdint>
#include <vector>

void compute_eigenvalues_and_vectors(int64_t n, const std::vector<double> &A,
                                     std::vector<double> &W,
                                     std::vector<double> &VR);

#endif // COMPUTE_EIGENVALUES_H