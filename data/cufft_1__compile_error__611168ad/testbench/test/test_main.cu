#include "fft_mgpu.h"
#include "fft_helpers.h"
#include <complex>
#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

// Forward declaration
void single(int fft_size, int batch_size,
            std::vector<std::complex<float>> &h_data_in,
            std::vector<std::complex<float>> &h_data_out);

static void run_benchmark() {
  int fft_size = 65536;
  int batch_size = 16;
  std::vector<int> gpus = {0, 0};
  size_t element_count = static_cast<size_t>(fft_size) * batch_size;

  std::vector<std::complex<float>> data_in(element_count);
  std::vector<std::complex<float>> data_out(element_count);

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (size_t i = 0; i < data_in.size(); ++i) {
    data_in[i] = {dis(gen), dis(gen)};
  }

  cufftXtSubFormat_t decomposition = CUFFT_XT_FORMAT_INPLACE;

  // Warmup
  for (int i = 0; i < 3; ++i) {
    fft_1d_mgpu_c2c_example(fft_size, batch_size, gpus, data_in, data_out,
                            decomposition);
  }
  cudaDeviceSynchronize();

  // Timed region
  nvtxRangePushA("bench_region");
  for (int i = 0; i < 100; ++i) {
    fft_1d_mgpu_c2c_example(fft_size, batch_size, gpus, data_in, data_out,
                            decomposition);
  }
  cudaDeviceSynchronize();
  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  bool perf = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--perf") == 0) {
      perf = true;
    }
  }

  if (perf) {
    run_benchmark();
    return 0;
  }

  auto test_spmg = []() {
    int fft_size = 256;
    std::vector<int> gpus = {0, 0};
    int batch_size = 1;

    size_t element_count = fft_size * batch_size;

    std::vector<std::complex<float>> data_in(element_count);
    std::vector<std::complex<float>> data_out_reference(element_count,
                                                        {-1.0f, -1.0f});
    std::vector<std::complex<float>> data_out(element_count, {-0.5f, -0.5f});

    std::mt19937 gen(3);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < data_in.size(); ++i) {
      float real = dis(gen);
      float imag = dis(gen);
      data_in[i] = {real, imag};
    }

    cufftXtSubFormat_t decomposition = CUFFT_XT_FORMAT_INPLACE;

    fft_1d_mgpu_c2c_example(fft_size, batch_size, gpus, data_in, data_out,
                            decomposition);
    single(fft_size, batch_size, data_in, data_out_reference);

    // Verify results
    double error = 0.0;
    double ref = 0.0;
    for (size_t i = 0; i < element_count; ++i) {
      error += std::norm(data_out[i] - data_out_reference[i]);
      ref += std::norm(data_out_reference[i]);
    }

    double l2_error =
        (ref == 0.0) ? std::sqrt(error) : std::sqrt(error) / std::sqrt(ref);
    if (l2_error < 0.001) {
      std::cout << "PASSED with L2 error = " << l2_error << std::endl;
    } else {
      std::cerr << "FAILED with L2 error = " << l2_error << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };

  test_spmg();
}