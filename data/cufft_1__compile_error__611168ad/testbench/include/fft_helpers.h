#ifndef FFT_HELPERS_H
#define FFT_HELPERS_H

#include <cuda_runtime.h>
#include <cufftXt.h>
#include <iostream>
#include <cstdlib>

#define CUDA_RT_CALL(call)                                                     \
  {                                                                            \
    cudaError_t cudaStatus = call;                                             \
    if (cudaSuccess != cudaStatus) {                                           \
      std::cerr << "ERROR: CUDA RT call \"" #call "\" failed with "            \
                << cudaGetErrorString(cudaStatus) << " (" << cudaStatus        \
                << ") at " << __FILE__ << ":" << __LINE__ << std::endl;        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#define CUFFT_CALL(call)                                                       \
  {                                                                            \
    cufftResult_t cufftStatus = call;                                          \
    if (CUFFT_SUCCESS != cufftStatus) {                                        \
      std::cerr << "ERROR: cuFFT call \"" #call "\" failed with error code "   \
                << cufftStatus << " at " << __FILE__ << ":" << __LINE__        \
                << std::endl;                                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#endif // FFT_HELPERS_H