#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/copy.h>
#include <thrust/remove.h>  // CUDA 13: remove_if.h was deprecated
#include <thrust/tuple.h>

struct PredicateFilterOp {
    __host__ __device__
    bool operator()(const thrust::tuple<int, int>& tuple) const {
        // The task says: "filter out elements... wherever the corresponding value in the predicate array is one."
        // Wait, the examples say: {8, 7, -7, -6, 2}, {0, 1, 1, 0, 1} -> {7, -7, 2}.
        // This means 1 = KEEP, 0 = REMOVE.
        // thrust::copy_if keeps elements where predicate is true.
        // thrust::remove_if removes elements where predicate is true.
        // The prompt text says "filter out elements... wherever... value... is one", 
        // but the examples show that when value is 1, it is KEPT.
        // I will follow the examples: 1 means keep, 0 means remove.
        // Since we want to use copy_if or similar, we return true for elements to keep.
        return thrust::get<1>(tuple) == 0; // For remove_if: return true if we want to REMOVE (0).
    }
};

void predicateBasedFilter(thrust::device_vector<int>& inpVector,
                         thrust::device_vector<int>& predicateVector,
                         thrust::device_vector<int>& outVector) {
    
    // Create a zip iterator to pair input and predicate
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(inpVector.begin(), predicateVector.begin()));
    auto end = thrust::make_zip_iterator(thrust::make_tuple(inpVector.end(), predicateVector.end()));

    // We need to store the result. Since outVector size is not pre-defined, 
    // and we are using Thrust, the most efficient way to filter is copy_if.
    
    // Custom predicate for copy_if: return true to KEEP.
    struct KeepOp {
        __host__ __device__
        bool operator()(const thrust::tuple<int, int>& tuple) const {
            return thrust::get<1>(tuple) == 1;
        }
    };

    // Temporary vector to hold the zipped pairs that passed the filter
    thrust::device_vector<thrust::tuple<int, int>> filteredPairs(inpVector.size());
    auto resultEnd = thrust::copy_if(begin, end, filteredPairs.begin(), KeepOp());

    // Now extract the first element of each tuple into outVector
    outVector.resize(thrust::distance(filteredPairs.begin(), resultEnd));
    
    struct ExtractFirst {
        __host__ __device__
        int operator()(const thrust::tuple<int, int>& tuple) const {
            return thrust::get<0>(tuple);
        }
    };

    thrust::transform(filteredPairs.begin(), resultEnd, outVector.begin(), ExtractFirst());
}