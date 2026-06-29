#include "mix_and_scramble.h"

#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cstddef>
#include <numeric>

struct scramble_index_functor {
    int n;
    int a;
    int b;

    __host__ __device__
    int operator()(int i) const {
        return (static_cast<long long>(a) * i + b) % n;
    }
};

void mix_and_scramble(thrust::device_vector<int> const& vec,
                      thrust::device_vector<int>& res) {
    const int n = static_cast<int>(vec.size());
    res.resize(vec.size());
    if (n == 0) return;

    int a = n > 1 ? (n / 2 + 1) : 1;
    while (std::gcd(a, n) != 1) ++a;

    int b = static_cast<int>((2654435761ull ^ static_cast<unsigned long long>(n) * 2246822519ull) % static_cast<unsigned long long>(n));

    scramble_index_functor f{n, a, b};

    auto first = thrust::make_counting_iterator<int>(0);
    auto map_first = thrust::make_transform_iterator(first, f);
    auto map_last = map_first + n;

    thrust::gather(map_first, map_last, vec.begin(), res.begin());
}