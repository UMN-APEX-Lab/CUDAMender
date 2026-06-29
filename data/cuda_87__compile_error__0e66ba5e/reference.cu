#include "negate_and_prefix_sum.h"
#include <thrust/transform_scan.h>

thrust::device_vector<int> negate_and_prefix_sum(const thrust::device_vector<int> &vec)
{
    thrust::device_vector<int> res(vec.size());
    thrust::transform_exclusive_scan(
        vec.begin(), vec.end(), res.begin(), [=] __host__ __device__(int x) { return -x; }, 0,
        thrust::plus<int>());

    return res;
}