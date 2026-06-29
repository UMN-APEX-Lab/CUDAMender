#include "fft_mgpu.h"
#include "fft_helpers.h"

void fft_3d_mgpu_r2c_c2r_example(std::array<int, 3> &dims,
                                 std::vector<int> &gpus,
                                 std::vector<std::complex<float>> &h_data_in,
                                 std::vector<std::complex<float>> &h_data_out) {
  cufftHandle plan_r2c{};
  cufftHandle plan_c2r{};
  CUFFT_CALL(cufftCreate(&plan_r2c));
  CUFFT_CALL(cufftCreate(&plan_c2r));

#if CUFFT_VERSION >= 10400
  cudaStream_t stream{};
  CUDA_RT_CALL(cudaStreamCreate(&stream));
  CUFFT_CALL(cufftSetStream(plan_r2c, stream));
  CUFFT_CALL(cufftSetStream(plan_c2r, stream));
#endif

  CUFFT_CALL(cufftXtSetGPUs(plan_r2c, gpus.size(), gpus.data()));
  CUFFT_CALL(cufftXtSetGPUs(plan_c2r, gpus.size(), gpus.data()));

  size_t workspace_sizes[gpus.size()];
  CUFFT_CALL(cufftMakePlan3d(plan_r2c, dims[0], dims[1], dims[2], CUFFT_R2C,
                             workspace_sizes));
  CUFFT_CALL(cufftMakePlan3d(plan_c2r, dims[0], dims[1], dims[2], CUFFT_C2R,
                             workspace_sizes));

  cudaLibXtDesc *indesc = nullptr;
  CUFFT_CALL(cufftXtMalloc(plan_r2c, &indesc, CUFFT_XT_FORMAT_INPLACE));
  CUFFT_CALL(cufftXtMemcpy(plan_r2c, reinterpret_cast<void *>(indesc),
                           reinterpret_cast<void *>(h_data_in.data()),
                           CUFFT_COPY_HOST_TO_DEVICE));

  CUFFT_CALL(cufftXtExecDescriptor(plan_r2c, indesc, indesc, CUFFT_FORWARD));

  float scale{2.f};
  scaleComplex(indesc, scale, h_data_out.size(), gpus.size());

  CUFFT_CALL(cufftXtExecDescriptor(plan_c2r, indesc, indesc, CUFFT_INVERSE));

  CUFFT_CALL(cufftXtMemcpy(
      plan_c2r, reinterpret_cast<void *>(h_data_out.data()),
      reinterpret_cast<void *>(indesc), CUFFT_COPY_DEVICE_TO_HOST));

  CUFFT_CALL(cufftXtFree(indesc));
  CUFFT_CALL(cufftDestroy(plan_r2c));
  CUFFT_CALL(cufftDestroy(plan_c2r));

#if CUFFT_VERSION >= 10400
  CUDA_RT_CALL(cudaStreamDestroy(stream));
#endif
}