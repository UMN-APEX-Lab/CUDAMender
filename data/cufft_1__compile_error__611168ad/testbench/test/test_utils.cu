#include "fft_helpers.h"
#include <complex>
#include <vector>
#include <cufftXt.h>

// Single GPU version of cuFFT plan for reference.
void single(int fft_size, int batch_size,
            std::vector<std::complex<float>> &h_data_in,
            std::vector<std::complex<float>> &h_data_out) {

  cufftHandle plan{};

  CUFFT_CALL(cufftCreate(&plan));

  size_t workspace_size;
  CUFFT_CALL(
      cufftMakePlan1d(plan, fft_size, CUFFT_C2C, batch_size, &workspace_size));

  void *d_data = nullptr;
  size_t datasize = h_data_in.size() * sizeof(std::complex<float>);

  CUDA_RT_CALL(cudaMalloc(&d_data, datasize));
  CUDA_RT_CALL(
      cudaMemcpy(d_data, h_data_in.data(), datasize, cudaMemcpyHostToDevice));

  CUFFT_CALL(cufftXtExec(plan, d_data, d_data, CUFFT_FORWARD));

  CUDA_RT_CALL(
      cudaMemcpy(h_data_out.data(), d_data, datasize, cudaMemcpyDeviceToHost));
  CUDA_RT_CALL(cudaFree(d_data));

  CUFFT_CALL(cufftDestroy(plan));
}