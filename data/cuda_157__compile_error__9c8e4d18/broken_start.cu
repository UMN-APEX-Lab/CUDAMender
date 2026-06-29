#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <cuda_runtime.h>

struct CountToCdfFunctor {
    float inv_size;
    __host__ __device__
    explicit CountToCdfFunctor(float inv) : inv_size(inv) {}

    __host__ __device__
    float operator()(const int x) const {
        return static_cast<float>(x) * inv_size;
    }
};

void calculatePixelCdf(thrust::device_vector<int> &srcImage, int size, cudaStream_t stream, thrust::device_vector<float> &output) {
    output.resize(size);
    if (size <= 0) {
        return;
    }

    auto policy = thrust::cuda::par.on(stream);

    thrust::device_vector<int> sortedImage = srcImage;
    thrust::sort(policy, sortedImage.begin(), sortedImage.end());

    thrust::device_vector<int> uniqueKeys(size);
    thrust::device_vector<int> counts(size);

    auto reduce_end = thrust::reduce_by_key(
        policy,
        sortedImage.begin(),
        sortedImage.end(),
        thrust::make_constant_iterator<int>(1),
        uniqueKeys.begin(),
        counts.begin()
    );

    int numUnique = static_cast<int>(reduce_end.first - uniqueKeys.begin());

    thrust::device_vector<int> cumulativeCounts(numUnique);
    thrust::inclusive_scan(
        policy,
        counts.begin(),
        counts.begin() + numUnique,
        cumulativeCounts.begin()
    );

    thrust::device_vector<float> cdfValues(numUnique);
    thrust::transform(
        policy,
        cumulativeCounts.begin(),
        cumulativeCounts.end(),
        cdfValues.begin(),
        CountToCdfFunctor(1.0f / static_cast<float>(size))
    );

    thrust::device_vector<float> lut(256);
    thrust::fill(policy, lut.begin(), lut.end(), 0.0f);

    thrust::scatter(
        policy,
        cdfValues.begin(),
        cdfValues.end(),
        uniqueKeys.begin(),
        lut.begin()
    );

    thrust::gather(
        policy,
        srcImage.begin(),
        srcImage.begin() + size,
        lut.begin(),
        output.begin()
    );
}