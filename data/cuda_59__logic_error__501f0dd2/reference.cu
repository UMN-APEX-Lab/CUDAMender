#include "has_negatives.h"
#include <thrust/logical.h>

struct is_negative
{
    __host__ __device__ bool operator()(int x) const { return x < 0; }
};

bool has_negatives(thrust::device_vector<int> const &vec)
{
    return thrust::any_of(vec.begin(), vec.end(), is_negative {});
}