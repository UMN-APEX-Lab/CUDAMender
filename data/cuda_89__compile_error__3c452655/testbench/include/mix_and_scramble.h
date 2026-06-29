#ifndef MIX_AND_SCRAMBLE_H
#define MIX_AND_SCRAMBLE_H

#include <thrust/device_vector.h>

void mix_and_scramble(thrust::device_vector<int> const &vec, thrust::device_vector<int> &res);

#endif // MIX_AND_SCRAMBLE_H