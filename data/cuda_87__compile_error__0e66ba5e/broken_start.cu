#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h>  // CUDA 13: top-level was deprecated
#include <thrust/functional.h>

struct Negate {
    __host__ __device__
    int operator()(const int& x) const {
        return -x;
    }
};

thrust::device_vector<int> negate_and_prefix_sum(const thrust::device_vector<int>& vec) {
    thrust::device_vector<int> result(vec.size());
    
    // Use a transform_iterator to negate elements on-the-fly
    // and thrust::exclusive_scan to compute the prefix sum.
    thrust::exclusive_scan(
        thrust::make_transform_iterator(vec.begin(), Negate()),
        vec.end(),
        result.begin()
    );
    
    return result;
}