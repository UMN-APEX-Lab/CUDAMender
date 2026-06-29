#include "fft_mgpu.h"
#include "fft_helpers.h"

void fft_1d_mgpu_c2c_example(int fft_size, int batch_size,
                             std::vector<int> &gpus,
                             std::vector<std::complex<float>> &h_data_in,
                             std::vector<std::complex<float>> &h_data_out,
                             cufftXtSubFormat_t subformat) {

  cufftHandle plan{};
  CUFFT_CALL(cufftCreate(&plan));

  CUFFT_CALL(cufftXtSetGPUs(plan, gpus.size(), gpus.data()));

  size_t workspace_sizes[gpus.size()];
  CUFFT_CALL(
      cufftMakePlan1d(plan, fft_size, CUFFT_C2C, batch_size, workspace_sizes));

  cudaLibXtDesc *indesc = nullptr;
  CUFFT_CALL(cufftXtMalloc(plan, &indesc, subformat));
  CUFFT_CALL(cufftXtMemcpy(plan, reinterpret_cast<void *>(indesc),
                           reinterpret_cast<void *>(h_data_in.data()),
                           CUFFT_COPY_HOST_TO_DEVICE));
  CUFFT_CALL(cufftXtExecDescriptor(plan, indesc, indesc, CUFFT_FORWARD));
  CUFFT_CALL(cufftXtMemcpy(plan, reinterpret_cast<void *>(h_data_out.data()),
                           reinterpret_cast<void *>(indesc),
                           CUFFT_COPY_DEVICE_TO_HOST));
  CUFFT_CALL(cufftXtFree(indesc));
  CUFFT_CALL(cufftDestroy(plan));
}