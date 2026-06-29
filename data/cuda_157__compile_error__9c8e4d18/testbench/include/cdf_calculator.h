#ifndef CDF_CALCULATOR_H
#define CDF_CALCULATOR_H

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

void calculatePixelCdf(thrust::device_vector<int> &srcImage, int size, cudaStream_t stream, thrust::device_vector<float> &output);

#endif // CDF_CALCULATOR_H