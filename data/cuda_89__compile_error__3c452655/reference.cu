#include "mix_and_scramble.h"
#include <thrust/shuffle.h>
#include <thrust/random.h>

void mix_and_scramble(thrust::device_vector<int> const &vec, thrust::device_vector<int> &res)
{
    thrust::random::default_random_engine rand;
    thrust::shuffle_copy(vec.begin(), vec.end(), res.begin(), rand);
}