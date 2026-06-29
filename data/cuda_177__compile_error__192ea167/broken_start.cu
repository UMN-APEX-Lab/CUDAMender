#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "include/histogram.h"
#include "include/histogram_helpers.h"

__global__ void prepareRGBData(uint8_t* input, uint8_t* output, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        uint8_t* pixel = &input[idx * 4];
        // We need to flatten the RGB channels into a single stream for CUB MultiHistogram
        // However, CUB DeviceHistogram::MultiHistogramEven expects an array of values
        // and an array of IDs mapping each value to a histogram.
        // Since we have 3 values per pixel, we can't simply use the input pointer.
        // But wait, the task requires MultiHistogramEven.
        // This means we need:
        // 1. An array of data values (R, G, B, R, G, B...)
        // 2. An array of histogram IDs (0, 1, 2, 0, 1, 2...)
    }
}

// The prepare kernel is actually needed to transform interleaved RGBA to 
// a format suitable for cub::DeviceHistogram::MultiHistogramEven.
__global__ void unpackRGBA(const uint8_t* input, uint8_t* output, int* ids, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        const uint8_t* pixel = &input[idx * 4];
        output[idx * 3 + 0] = pixel[0]; // R
        output[idx * 3 + 1] = pixel[1]; // G
        output[idx * 3 + 2] = pixel[2]; // B
        
        ids[idx * 3 + 0] = 0;
        ids[idx * 3 + 1] = 1;
        ids[idx * 3 + 2] = 2;
    }
}

void calcMultiHistogram(uint8_t* inputData_d, int** histogram_d,
                        int* numLevels_h, unsigned int* lowerLevel_h,
                        unsigned int* upperLevel_h, int numPixels,
                        cudaStream_t stream) {
    int totalElements = numPixels * 3;
    uint8_t* d_values = nullptr;
    int* d_ids = nullptr;
    cudaMallocAsync(&d_values, totalElements * sizeof(uint8_t), stream);
    cudaMallocAsync(&d_ids, totalElements * sizeof(int), stream);

    int blockSize = 256;
    int gridSize = (numPixels + blockSize - 1) / blockSize;
    unpackRGBA<<<gridSize, blockSize, 0, stream>>>(inputData_d, d_values, d_ids, numPixels);

    // CUB MultiHistogramEven parameters
    // We have 3 histograms (R, G, B)
    int numHistograms = 3;
    
    // For MultiHistogramEven, we need to provide the range for each histogram.
    // Since the API takes arrays for numBins, lowerBound, upperBound:
    int* d_numBins = nullptr;
    unsigned int* d_lowerBound = nullptr;
    unsigned int* d_upperBound = nullptr;

    cudaMallocAsync(&d_numBins, numHistograms * sizeof(int), stream);
    cudaMallocAsync(&d_lowerBound, numHistograms * sizeof(unsigned int), stream);
    cudaMallocAsync(&d_upperBound, numHistograms * sizeof(unsigned int), stream);

    cudaMemcpyAsync(d_numBins, numLevels_h, numHistograms * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_lowerBound, lowerLevel_h, numHistograms * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_upperBound, upperLevel_h, numHistograms * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);

    // Determine temporary storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    cub::DeviceHistogram::MultiHistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_values, d_ids, totalElements,
        d_numBins, d_lowerBound, d_upperBound,
        (uint32_t*)histogram_d[0], // This is tricky: CUB expects a contiguous block or specific offsets
        numHistograms, 0, stream
    );

    // The above call is for sizing. But histogram_d is int**, meaning pointers to separate arrays.
    // cub::DeviceHistogram::MultiHistogramEven expects the output to be a contiguous array 
    // where each histogram starts at offset (numBins[i] * sizeof(int)).
    // Since the task provides int** histogram_d, we must handle the output carefully.
    // However, the standard CUB MultiHistogramEven writes to a single contiguous buffer.
    // To use the provided int** histogram_d, we can't use MultiHistogramEven directly 
    // unless we manage the offsets.
    
    // Correct approach for MultiHistogramEven with provided int**:
    // 1. Calculate total bins needed.
    // 2. Use a temporary contiguous buffer for output.
    // 3. Copy results from contiguous buffer to the pointers in histogram_d.

    size_t totalBins = 0;
    for(int i=0; i<numHistograms; ++i) totalBins += numLevels_h[i];
    int* d_out_contiguous = nullptr;
    cudaMallocAsync(&d_out_contiguous, totalBins * sizeof(int), stream);

    cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);

    cub::DeviceHistogram::MultiHistogramEven(
        d_temp_storage, temp_storage_bytes,
        d_values, d_ids, totalElements,
        d_numBins, d_lowerBound, d_upperBound,
        d_out_contiguous,
        numHistograms, 0, stream
    );

    // Copy contiguous output to the provided int** histogram_d
    int offset = 0;
    for (int i = 0; i < numHistograms; ++i) {
        cudaMemcpyAsync(histogram_d[i], &d_out_contiguous[offset], numLevels_h[i] * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        offset += numLevels_h[i];
    }

    cudaFreeAsync(d_values, stream);
    cudaFreeAsync(d_ids, stream);
    cudaFreeAsync(d_numBins, stream);
    cudaFreeAsync(d_lowerBound, stream);
    cudaFreeAsync(d_upperBound, stream);
    cudaFreeAsync(d_temp_storage, stream);
    cudaFreeAsync(d_out_contiguous, stream);
}