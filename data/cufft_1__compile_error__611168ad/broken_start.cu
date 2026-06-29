#include "include/fft_mgpu.h"
#include "include/fft_helpers.h"

#include <cufftXt.h>
#include <cuda_runtime.h>

#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CHECK_CUDA
#if defined(CUDA_RT_CALL)
#define CHECK_CUDA(expr) CUDA_RT_CALL(expr)
#elif defined(CUDA_CHECK)
#define CHECK_CUDA(expr) CUDA_CHECK(expr)
#else
#define CHECK_CUDA(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err)); \
        }                                                                                \
    } while (0)
#endif
#endif

#ifndef CHECK_CUFFT
#if defined(CUFFT_CALL)
#define CHECK_CUFFT(expr) CUFFT_CALL(expr)
#elif defined(CUFFT_CHECK)
#define CHECK_CUFFT(expr) CUFFT_CHECK(expr)
#else
#define CHECK_CUFFT(expr)                                                                \
    do {                                                                                 \
        cufftResult _err = (expr);                                                       \
        if (_err != CUFFT_SUCCESS) {                                                     \
            throw std::runtime_error("cuFFT error");                                     \
        }                                                                                \
    } while (0)
#endif
#endif

void fft_1d_mgpu_c2c_example(int fft_size, int batch_size,
                             std::vector<int> &gpus,
                             std::vector<std::complex<float>> &h_data_in,
                             std::vector<std::complex<float>> &h_data_out,
                             cufftXtSubFormat_t subformat) {
    if (fft_size <= 0 || batch_size <= 0) {
        throw std::invalid_argument("fft_size and batch_size must be positive");
    }
    if (gpus.empty()) {
        throw std::invalid_argument("gpus must not be empty");
    }

    const size_t elem_count = static_cast<size_t>(fft_size) * static_cast<size_t>(batch_size);
    if (h_data_in.size() < elem_count) {
        throw std::invalid_argument("h_data_in size is smaller than fft_size * batch_size");
    }
    h_data_out.resize(elem_count);

    cufftHandle plan = 0;
    cudaLibXtDesc *device_desc = nullptr;

    CHECK_CUFFT(cufftCreate(&plan));
    try {
        CHECK_CUFFT(cufftXtSetGPUs(plan, static_cast<int>(gpus.size()), gpus.data()));

        std::vector<size_t> work_sizes(gpus.size(), 0);
        CHECK_CUFFT(cufftMakePlan1d(plan, fft_size, CUFFT_C2C, batch_size, work_sizes.data()));

        CHECK_CUFFT(cufftXtMalloc(plan, &device_desc, subformat));

        CHECK_CUFFT(cufftXtMemcpy(plan,
                                  device_desc,
                                  reinterpret_cast<void *>(h_data_in.data()),
                                  CUFFT_COPY_HOST_TO_DEVICE));

        CHECK_CUFFT(cufftXtExecDescriptorC2C(plan, device_desc, device_desc, CUFFT_FORWARD));

        CHECK_CUFFT(cufftXtMemcpy(plan,
                                  reinterpret_cast<void *>(h_data_out.data()),
                                  device_desc,
                                  CUFFT_COPY_DEVICE_TO_HOST));

        CHECK_CUFFT(cufftXtFree(device_desc));
        device_desc = nullptr;

        CHECK_CUFFT(cufftDestroy(plan));
    } catch (...) {
        if (device_desc != nullptr) {
            cufftXtFree(device_desc);
        }
        if (plan != 0) {
            cufftDestroy(plan);
        }
        throw;
    }
}