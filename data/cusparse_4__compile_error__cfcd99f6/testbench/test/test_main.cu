#include "gather.h"
#include "cuda_helpers.h"
#include <vector>
#include <iostream>
#include <cstring>
#include <numeric>
#include <nvtx3/nvToolsExt.h>

int main(int argc, char* argv[]) {
bool perf = (argc > 1 && std::strcmp(argv[1], "--perf") == 0);

if (perf) {
  const int size = 10'000'000;
  const int nnz  = 5'000'000;

  std::vector<int> hX_indices(nnz);
  for (int i = 0; i < nnz; i++) hX_indices[i] = i * 2;

  std::vector<float> hY(size);
  std::iota(hY.begin(), hY.end(), 1.0f);

  std::vector<float> hX_values(nnz);

  for (int w = 0; w < 3; w++) {
    gather(size, nnz, hX_indices, hY, hX_values);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < 100; i++) {
    gather(size, nnz, hX_indices, hY, hX_values);
  }
  nvtxRangePop();

  return 0;
}
auto test_gather = []() {
  const int size = 8;
  const int nnz = 4;
  const std::vector<int> hX_indices = {0, 3, 4, 7};
  const std::vector<float> hY = {1.0f, 2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f};
  const std::vector<float> hX_result = {1.0f, 4.0f, 5.0f, 8.0f};

  std::vector<float> hX_values(nnz);

  gather(size, nnz, hX_indices, hY, hX_values);

  bool correct = true;
  for (int i = 0; i < nnz; i++) {
    if (hX_values[i] != hX_result[i]) {
      correct = false;
      break;
    }
  }

  if (correct) {
    std::cout << "gather test PASSED" << std::endl;
  } else {
    std::cout << "gather test FAILED: wrong result" << std::endl;
    std::exit(EXIT_FAILURE);
  }
};

test_gather();

}