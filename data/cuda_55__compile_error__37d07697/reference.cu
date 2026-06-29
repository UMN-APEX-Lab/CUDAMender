#include "local_prefix_sum.h"
#include <thrust/scan.h>

thrust::device_vector<int> local_prefix_exclusive_sum(thrust::device_vector<int> const &vec,
                                                      thrust::device_vector<int> const &keys)
{
    thrust::device_vector<int> result(vec.size());
    thrust::exclusive_scan_by_key(keys.begin(), keys.end(), vec.begin(), result.begin());
    return result;
}