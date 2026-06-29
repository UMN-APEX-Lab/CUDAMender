#ifndef FFT_MGPU_H
#define FFT_MGPU_H

#include <complex>
#include <vector>
#include <cufftXt.h>

void fft_1d_mgpu_c2c_example(int fft_size, int batch_size,
                             std::vector<int> &gpus,
                             std::vector<std::complex<float>> &h_data_in,
                             std::vector<std::complex<float>> &h_data_out,
                             cufftXtSubFormat_t subformat);

#endif // FFT_MGPU_H