#include "has_negatives.h"

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <thrust/execution_policy.h>

// Predicate used by the thrust algorithm to flag elements of interest.
// Each element is examined in turn; the algorithm returns true on the
// first element for which this functor evaluates to true.
struct sign_pred {
    static constexpr int threshold = 0;

    __host__ __device__
    bool operator()(int x) const
    {
        return x > threshold;
    }
};

// Returns true iff the input vector contains an element of the
// signed type that the predicate above flags. Uses thrust::any_of
// so the algorithm short-circuits and runs on the device.
bool has_negatives(thrust::device_vector<int> const& vec)
{
    return thrust::any_of(
        thrust::device,
        vec.begin(),
        vec.end(),
        sign_pred{}
    );
}
