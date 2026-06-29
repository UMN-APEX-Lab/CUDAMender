#ifndef TRTRI_H
#define TRTRI_H

#include <vector>
#include <cublas_v2.h>

void trtri(int n, std::vector<double>& A, cublasFillMode_t uplo,
           cublasDiagType_t diag);

#endif // TRTRI_H