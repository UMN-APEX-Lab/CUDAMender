#ifndef LOCAL_PREFIX_SUM_H
#define LOCAL_PREFIX_SUM_H

#include <thrust/device_vector.h>

thrust::device_vector<int> local_prefix_exclusive_sum(thrust::device_vector<int> const &vec,
                                                      thrust::device_vector<int> const &keys);

#endif // LOCAL_PREFIX_SUM_H