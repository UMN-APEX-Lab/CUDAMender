#include "scale_vector.h"
#include <cmath>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

void run_correctness_test() {
  const int n = 4;
  const double alpha = 2.2;
  std::vector<double> h_in = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> h_out(n);

  scale_vector(n, alpha, h_in, h_out);

  for (int i = 0; i < n; ++i) {
    if (std::abs(h_out[i] - h_in[i] * alpha) > 1e-6) {
      std::cerr << "Test failed at index " << i << ": expected "
                << h_in[i] * alpha << ", got " << h_out[i] << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  std::cout << "Test passed!" << std::endl;
}

void run_benchmark() {
  const int n = 4 * 1024 * 1024;
  const double alpha = 1.5;
  std::vector<double> h_in(n);
  std::vector<double> h_out(n);
  for (int i = 0; i < n; ++i) h_in[i] = static_cast<double>(i);

  const int warmup_iters = 3;
  const int timed_iters = 100;

  for (int i = 0; i < warmup_iters; ++i) {
    scale_vector(n, alpha, h_in, h_out);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed_iters; ++i) {
    scale_vector(n, alpha, h_in, h_out);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
  } else {
    run_correctness_test();
  }
  return 0;
}