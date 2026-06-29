#ifndef FFT_MGPU_H
#define FFT_MGPU_H

#include <array>
#include <complex>
#include <vector>

void fft_3d_mgpu_r2c_c2r_example(std::array<int, 3> &dims,
                                 std::vector<int> &gpus,
                                 std::vector<std::complex<float>> &h_data_in,
                                 std::vector<std::complex<float>> &h_data_out);

#endif // FFT_MGPU_H