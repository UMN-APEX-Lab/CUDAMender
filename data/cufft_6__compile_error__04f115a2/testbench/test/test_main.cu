#include "fft_mgpu.h"
#include "fft_helpers.h"
#include <array>
#include <complex>
#include <cstring>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <random>
#include <vector>

// Single GPU version of cuFFT plan for reference.
void single(std::array<int, 3> &dims,
            std::vector<std::complex<float>> &h_data_in,
            std::vector<std::complex<float>> &h_data_out) {

  // Initiate cufft plans, one for r2c and one for c2r
  cufftHandle plan_r2c{};
  cufftHandle plan_c2r{};
  CUFFT_CALL(cufftCreate(&plan_r2c));
  CUFFT_CALL(cufftCreate(&plan_c2r));

  // Create the plans
  size_t workspace_size;
  CUFFT_CALL(cufftMakePlan3d(plan_r2c, dims[0], dims[1], dims[2], CUFFT_R2C,
                             &workspace_size));
  CUFFT_CALL(cufftMakePlan3d(plan_c2r, dims[0], dims[1], dims[2], CUFFT_C2R,
                             &workspace_size));

  void *d_data = nullptr;
  size_t datasize = h_data_in.size() * sizeof(std::complex<float>);

  // Copy input data to GPUs
  CUDA_RT_CALL(cudaMalloc(&d_data, datasize));
  CUDA_RT_CALL(
      cudaMemcpy(d_data, h_data_in.data(), datasize, cudaMemcpyHostToDevice));

  // Execute the plan_r2c
  CUFFT_CALL(cufftXtExec(plan_r2c, d_data, d_data, CUFFT_FORWARD));

  // Scale complex results
  float scale{2.f};
  int threads{1024};

  int dimGrid = (h_data_in.size() + threads - 1) / threads;
  int dimBlock = threads;
  scaling_kernel<<<dimGrid, dimBlock>>>(
      reinterpret_cast<cufftComplex *>(d_data), h_data_in.size(), scale);

  // Execute the plan_c2r
  CUFFT_CALL(cufftXtExec(plan_c2r, d_data, d_data, CUFFT_INVERSE));

  // Copy output data to CPU
  CUDA_RT_CALL(
      cudaMemcpy(h_data_out.data(), d_data, datasize, cudaMemcpyDeviceToHost));
  CUDA_RT_CALL(cudaFree(d_data));

  CUFFT_CALL(cufftDestroy(plan_r2c));
  CUFFT_CALL(cufftDestroy(plan_c2r));
}

void run_benchmark() {
  std::array<int, 3> dims = {128, 128, 128};
  std::vector<int> gpus = {0, 0};

  size_t element_count = dims[0] * dims[1] * ((dims[2] / 2) + 1);

  std::vector<std::complex<float>> data_in(element_count);
  std::vector<std::complex<float>> data_out(element_count, {-1.0f, -1.0f});

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  for (size_t i = 0; i < data_in.size(); ++i) {
    data_in[i] = {dis(gen), dis(gen)};
  }

  const int warmup_iters = 3;
  const int timed_iters = 100;

  for (int i = 0; i < warmup_iters; ++i) {
    fft_3d_mgpu_r2c_c2r_example(dims, gpus, data_in, data_out);
  }

  nvtxRangePushA("bench_region");
  for (int i = 0; i < timed_iters; ++i) {
    fft_3d_mgpu_r2c_c2r_example(dims, gpus, data_in, data_out);
  }
  nvtxRangePop();
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::strcmp(argv[1], "--perf") == 0) {
    run_benchmark();
    return 0;
  }

  auto test_fft_3d_mgpu_r2c_c2r_example = []() {
    std::array<int, 3> dims = {64, 64, 64};
    std::vector<int> gpus = {0, 0};

    size_t element_count = dims[0] * dims[1] * ((dims[2] / 2) + 1);

    std::vector<std::complex<float>> data_in(element_count);
    std::vector<std::complex<float>> data_out(element_count, {-1.0f, -1.0f});
    std::vector<std::complex<float>> data_out_reference(element_count,
                                                        {-1.0f, -1.0f});

    std::mt19937 gen(3);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < data_in.size(); ++i) {
      float real = dis(gen);
      float imag = dis(gen);
      data_in[i] = {real, imag};
    }

    fft_3d_mgpu_r2c_c2r_example(dims, gpus, data_in, data_out);
    single(dims, data_in, data_out_reference);

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

  test_fft_3d_mgpu_r2c_c2r_example();
}