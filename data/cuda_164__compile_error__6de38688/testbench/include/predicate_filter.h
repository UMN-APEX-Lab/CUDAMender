#ifndef PREDICATE_FILTER_H
#define PREDICATE_FILTER_H

#include <thrust/device_vector.h>

void predicateBasedFilter(thrust::device_vector<int>& inpVector, 
                         thrust::device_vector<int>& predicateVector, 
                         thrust::device_vector<int>& outVector);

#endif // PREDICATE_FILTER_H