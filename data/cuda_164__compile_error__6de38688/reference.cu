#include "predicate_filter.h"
#include <thrust/reduce.h>
#include <thrust/partition.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

void predicateBasedFilter(thrust::device_vector<int>& inOutVector, thrust::device_vector<int>& predicateVector, thrust::device_vector<int>& outVector) {

	// Compute output length using the predicate vector and filter the output
	int outLength = thrust::reduce(thrust::device, predicateVector.begin(), predicateVector.end());
	thrust::stable_partition(thrust::device, inOutVector.begin(), inOutVector.end(), predicateVector.begin(), ::cuda::std::identity{});

	// Copy subarray from device memory to host memory
	outVector.resize(outLength);
	thrust::copy(thrust::device, inOutVector.begin(), inOutVector.begin() + outLength, outVector.begin());
}