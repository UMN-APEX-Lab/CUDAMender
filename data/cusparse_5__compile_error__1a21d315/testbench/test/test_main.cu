#include "gpsv_interleaved_batch.h"
#include "error_checks.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <nvtx3/nvToolsExt.h>

static void run_correctness_test() {
  const int n = 4;
  const int batchSize = 2;
  const int full_size = n * batchSize;

  std::vector<float> h_S = {0, 0, 11, 12, 0, 0, 25, 26};
  std::vector<float> h_L = {0, 5, 6, 7, 0, 19, 20, 21};
  std::vector<float> h_M = {1, 2, 3, 4, 15, 16, 17, 18};
  std::vector<float> h_U = {8, 9, 10, 0, 22, 23, 24, 0};
  std::vector<float> h_W = {13, 14, 0, 0, 27, 28, 0, 0};
  std::vector<float> h_B = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> h_X(full_size, 0.0f);

  solveGpsvInterleavedBatch(n, batchSize, h_S, h_L, h_M, h_U, h_W, h_B, h_X);

  std::cout << "==== x1 = inv(A1)*b1" << std::endl;
  for (int j = 0; j < n; j++)
    std::cout << "x1[" << j << "] = " << h_X[j] << std::endl;

  std::cout << "\n==== x2 = inv(A2)*b2" << std::endl;
  for (int j = 0; j < n; j++)
    std::cout << "x2[" << j << "] = " << h_X[n + j] << std::endl;
}

static void run_benchmark() {
  const int n = 512;
  const int batchSize = 1024;
  const int full_size = n * batchSize;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> diag_dist(5.0f, 10.0f);

  std::vector<float> h_S(full_size);
  std::vector<float> h_L(full_size);
  std::vector<float> h_M(full_size);
  std::vector<float> h_U(full_size);
  std::vector<float> h_W(full_size);
  std::vector<float> h_B(full_size);
  std::vector<float> h_X(full_size, 0.0f);

  for (int b = 0; b < batchSize; b++) {
    for (int i = 0; i < n; i++) {
      int idx = b * n + i;
      h_M[idx] = diag_dist(rng);
      h_L[idx] = (i >= 1) ? dist(rng) : 0.0f;
      h_U[idx] = (i < n - 1) ? dist(rng) : 0.0f;
      h_S[idx] = (i >= 2) ? dist(rng) : 0.0f;
      h_W[idx] = (i < n - 2) ? dist(rng) : 0.0f;
      h_B[idx] = dist(rng);
    }
  }

  const int warmup_iters = 3;
  const int timed_iters = 100;

  for (int i = 0; i < warmup_iters; i++) {
    solveGpsvInterleavedBatch(n, batchSize, h_S, h_L, h_M, h_U, h_W, h_B, h_X);
  }
  cudaDeviceSynchronize();

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed_iters; i++) {
    solveGpsvInterleavedBatch(n, batchSize, h_S, h_L, h_M, h_U, h_W, h_B, h_X);
  }
  cudaDeviceSynchronize();
  nvtxRangePop();
}

int main(int argc, char* argv[]) {
  bool perf = false;
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--perf") == 0) perf = true;
  }

  if (perf) {
    run_benchmark();
  } else {
    run_correctness_test();
  }

  return 0;
}