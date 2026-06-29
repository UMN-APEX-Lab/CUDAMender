#include "square_indices.h"
#include <thrust/tabulate.h>

void square_indices(thrust::device_vector<int> &vec)
{
    return thrust::tabulate(vec.begin(), vec.end(), [] __device__(int i) { return i * i; });
}