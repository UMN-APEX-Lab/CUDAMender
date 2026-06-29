#include "fft_helpers.h"

__global__ void scaling_kernel(cufftComplex *data, size_t size, float scale) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

void scaleComplex(cudaLibXtDesc *desc, const float scale, const size_t N,
                  const int nGPUs) {
  int device;
  int threads{1024};

  int dimGrid = (N / nGPUs + threads - 1) / threads;
  int dimBlock = threads;

  for (int i = 0; i < nGPUs; i++) {
    device = desc->descriptor->GPUs[i];
    CUDA_RT_CALL(cudaSetDevice(device));

    scaling_kernel<<<dimGrid, dimBlock>>>(
        (cufftComplex *)desc->descriptor->data[i],
        desc->descriptor->size[i] / sizeof(cufftComplex), scale);
    if (N / nGPUs != desc->descriptor->size[i] / sizeof(cufftComplex)) {
      throw std::runtime_error("ERROR: Wrong data size");
    }
  }

  for (int i = 0; i < nGPUs; i++) {
    device = desc->descriptor->GPUs[i];
    CUDA_RT_CALL(cudaSetDevice(device));
    CUDA_RT_CALL(cudaDeviceSynchronize());
  }
}