#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include "square_indices.h"

struct square_op {
    __host__ __device__
    int operator()(int x) const {
        return x * x;
    }
};

void square_indices(thrust::device_vector<int> &vec) {
    thrust::transform(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(static_cast<int>(vec.size())),
        vec.begin(),
        square_op()
    );
}