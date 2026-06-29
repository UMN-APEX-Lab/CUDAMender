#include <cub/cub.cuh>
#include "histogram.h"
#include "histogram_helpers.h"

void calcMultiHistogram(uint8_t* inputImage_d, int** histogram_d,
                          int* numLevels_h, unsigned int* lowerLevel_h,
                          unsigned int* upperLevel_h, int numPixels,
                          cudaStream_t stream) {

    // Determine temporary device storage requirements
    void* workStorage_d  = nullptr;
    size_t   workStorageBytes = 0;
    CUB_CHECK((cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        workStorage_d, workStorageBytes,
      inputImage_d, histogram_d, numLevels_h,
      lowerLevel_h, upperLevel_h, numPixels, stream)));

    // Allocate temporary storage
    CUDA_CHECK(cudaMallocAsync(&workStorage_d, workStorageBytes, stream));

    // Compute histograms
    CUB_CHECK((cub::DeviceHistogram::MultiHistogramEven<NUM_CHANNELS, NUM_ACTIVE_CHANNELS>(
        workStorage_d, workStorageBytes,
      inputImage_d, histogram_d, numLevels_h,
      lowerLevel_h, upperLevel_h, numPixels, stream)));

    CUDA_CHECK(cudaFreeAsync(workStorage_d, stream));
}