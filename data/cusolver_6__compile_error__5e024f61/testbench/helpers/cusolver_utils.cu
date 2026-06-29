#include "lu_factorization.h"

void printMatrix(int m, int n, const double *A, int lda, const char *name) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < n; col++) {
      printf("%s(%d,%d) = %f\n", name, row + 1, col + 1, A[row + col * lda]);
    }
  }
}