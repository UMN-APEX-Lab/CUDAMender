#ifndef HAS_NEGATIVES_H
#define HAS_NEGATIVES_H

#include <thrust/device_vector.h>

bool has_negatives(thrust::device_vector<int> const &vec);

#endif // HAS_NEGATIVES_H