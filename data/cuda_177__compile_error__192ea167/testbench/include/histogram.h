#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <cstdint>
#include <cuda_runtime.h>

void calcMultiHistogram(uint8_t* inputData_d, int** histogram_d,
                        int* numLevels_h, unsigned int* lowerLevel_h,
                        unsigned int* upperLevel_h, int numPixels,
                        cudaStream_t stream);

#endif // HISTOGRAM_H