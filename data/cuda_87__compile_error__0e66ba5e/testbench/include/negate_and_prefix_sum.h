#ifndef NEGATE_AND_PREFIX_SUM_H
#define NEGATE_AND_PREFIX_SUM_H

#include <thrust/device_vector.h>

thrust::device_vector<int> negate_and_prefix_sum(const thrust::device_vector<int> &vec);

#endif // NEGATE_AND_PREFIX_SUM_H