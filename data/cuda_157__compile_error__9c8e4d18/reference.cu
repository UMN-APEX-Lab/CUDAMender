#include "cdf_calculator.h"
#include "cuda_utils.h"
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

void calculatePixelCdf(thrust::device_vector<int> &srcImage_d, int size, cudaStream_t stream, thrust::device_vector<float> &output) {
    // Creating local device vectors
    thrust::device_vector<int> keys_d(size);
    thrust::device_vector<int> counts_d(size);
    thrust::device_vector<float> result_d(size);
    thrust::device_vector<int> sortedImage_d(size);
    // constant device vector of 1s for counting
    thrust::device_vector<int> constantOnes_d(size, 1);

    // Copy data into another device array
    CUDA_CHECK(cudaMemcpyAsync( thrust::raw_pointer_cast(sortedImage_d.data()),
                                thrust::raw_pointer_cast(srcImage_d.data()),
                                size * sizeof(int), cudaMemcpyDeviceToDevice, stream));

    // Sort the data
    thrust::sort(thrust::cuda::par.on(stream), sortedImage_d.begin(), sortedImage_d.end());

    // Count occurrences on histogram using reduce_by_key
    auto key_count_end = thrust::reduce_by_key(thrust::cuda::par.on(stream), sortedImage_d.begin(),
                                     sortedImage_d.end(), constantOnes_d.begin(),
                                     keys_d.begin(), counts_d.begin());

    int numUniqueKeys = key_count_end.first - keys_d.begin();
    keys_d.resize(numUniqueKeys);
    counts_d.resize(numUniqueKeys);

    // Normalize histogram to get frequencies
    thrust::device_vector<float> freq_d(numUniqueKeys);
    float normFactor = 1.0f / size;
    thrust::transform(
        thrust::cuda::par.on(stream),
        counts_d.begin(), counts_d.end(),
        freq_d.begin(),
        [=] __device__ (int count) {
            return count * normFactor;
        });

    // Compute cumulative sum of frequencies
    thrust::inclusive_scan(thrust::cuda::par.on(stream), freq_d.begin(), freq_d.end(), freq_d.begin());

    // Creating LUT with 256 entries
    thrust::device_vector<float> lut_d(256, 0.0f);
    thrust::scatter(
        thrust::cuda::par.on(stream),
        freq_d.begin(), freq_d.end(), //freq_d.begin() + numUniqueKeys,
        keys_d.begin(),
        lut_d.begin()
    );

    // Map each pixel to its CDF using LUT
    float* lut_ptr = thrust::raw_pointer_cast(lut_d.data());
    thrust::transform(
        thrust::cuda::par.on(stream),
        srcImage_d.begin(), srcImage_d.end(),
        output.begin(),
        [=] __device__ (int pixel) {
            return lut_ptr[pixel];
        });
}