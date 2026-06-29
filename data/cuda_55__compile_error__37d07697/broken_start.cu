#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>

/**
 * The goal is to compute a local exclusive prefix sum.
 * For each segment of identical keys, the sum resets to 0.
 * 
 * Algorithm:
 * 1. Compute the global exclusive prefix sum of the input vector.
 * 2. Identify the start of each segment (where keys[i] != keys[i-1]).
 * 3. For each element, subtract the global prefix sum value that existed 
 *    just before the start of its segment.
 */

thrust::device_vector<int> local_prefix_exclusive_sum(
    thrust::device_vector<int> const &vec,
    thrust::device_vector<int> const &keys) {
    
    size_t n = vec.size();
    if (n == 0) return thrust::device_vector<int>();

    // 1. Global exclusive prefix sum
    thrust::device_vector<int> global_sum(n);
    thrust::exclusive_scan(thrust::device, vec.begin(), vec.end(), global_sum.begin());

    // 2. Identify segment boundaries and store the offset
    // offset[i] will store the global_sum value at the start of the segment containing index i.
    thrust::device_vector<int> offsets(n);
    
    // We need to find the value of global_sum[i] where i is the first index of a segment.
    // Let's create a mask where 1 indicates the start of a new segment.
    thrust::device_vector<int> is_start(n, 0);
    is_start[0] = 1;
    
    // Use adjacent_difference or a custom transform to mark starts
    // keys[i] != keys[i-1]
    if (n > 1) {
        thrust::transform(thrust::device, 
                         keys.begin() + 1, keys.end(), 
                         keys.begin(), 
                         is_start.begin() + 1, 
                         [] __device__ (int curr, int prev) {
                             return (curr != prev) ? 1 : 0;
                         });
    }

    // Now we want to propagate the global_sum value at the start of the segment across the segment.
    // We can use a custom operation or simply: 
    // value_at_start[i] = (is_start[i]) ? global_sum[i] : 0
    thrust::device_vector<int> start_vals(n);
    thrust::transform(thrust::device, 
                      is_start.begin(), is_start.end(), 
                      global_sum.begin(), 
                      start_vals.begin(), 
                      [] __device__ (int start, int val) {
                          return start ? val : 0;
                      });

    // Now we use an inclusive scan with a "max" or "last non-zero" to fill the offsets.
    // Since global_sum is non-decreasing (if elements were non-negative), 
    // but elements can be negative, we need a way to carry the last start_val.
    // A simpler way: use thrust::inclusive_scan with a custom operator that replaces 0 with the previous value.
    
    struct Propagate {
        __device__ int operator()(int a, int b) const {
            return (b != 0) ? b : a;
        }
    };
    
    // However, global_sum[0] could be 0. We need to handle that.
    // Let's use a more robust approach:
    // The offset for index i is global_sum[first_index_of_segment].
    // We can use a scan to find the index of the start of the segment.
    thrust::device_vector<int> segment_id(n);
    thrust::inclusive_scan(thrust::device, is_start.begin(), is_start.end(), segment_id.begin());
    
    // Now we need the global_sum at the first index where segment_id == current_id.
    // This is tricky. Let's use the "inclusive scan with replace" logic but 
    // we need to distinguish between "value is 0" and "no value provided".
    
    // Alternative: 
    // 1. Find indices of starts.
    // 2. Store global_sum at those indices.
    // 3. Use search_sorted or similar to map each index to its start_val.
    // But we can just use a simple kernel.
    
    thrust::device_vector<int> result(n);
    
    // To avoid writing a custom kernel, we can use the following:
    // The offset for element i is global_sum[index of first element of segment].
    // Let's calculate the global_sum of the elements *before* the current segment.
    
    // We want: result[i] = global_sum[i] - global_sum[start_of_segment]
    // Let's find the global_sum at the start of each segment.
    thrust::device_vector<int> segment_starts_val(n);
    thrust::transform(thrust::device, 
                      is_start.begin(), is_start.end(), 
                      global_sum.begin(), 
                      segment_starts_val.begin(), 
                      [] __device__ (int start, int val) {
                          return start ? val : 0; 
                      });
    
    // To propagate the value, we use a scan. Since we need to handle 0s, 
    // we can't just use a simple sum. We use a custom binary operator.
    // Since we know the segments are contiguous, we can use a custom kernel.
    
    auto propagate_kernel = [] __global__ (int* out, const int* starts, const int* is_start, int n) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n) return;
        
        // This is a naive approach; for a real CUDA kernel we'd use shared memory/warp shuffles.
        // But for this task, a simple loop or thrust-based approach is expected.
        // Given the constraints, let's use a thrust-friendly approach.
    };

    // Correct Thrust approach to propagate:
    // Use a scan to find the index of the start of the segment for each element.
    thrust::device_vector<int> start_indices(n);
    thrust::device_vector<int> indices(n);
    thrust::sequence(indices.begin(), indices.end());
    
    thrust::transform(thrust::device, 
                      is_start.begin(), is_start.end(), 
                      indices.begin(), 
                      start_indices.begin(), 
                      [] __device__ (int start, int idx) {
                          return start ? idx : -1;
                      });
    
    // Propagate the index
    struct MaxOp {
        __device__ int operator()(int a, int b) const {
            return (b != -1) ? b : a;
        }
    };
    thrust::inclusive_scan(thrust::device, start_indices.begin(), start_indices.end(), start_indices.begin(), MaxOp());
    
    // Now result[i] = global_sum[i] - global_sum[start_indices[i]]
    thrust::transform(thrust::device, 
                      global_sum.begin(), global_sum.end(), 
                      start_indices.begin(), 
                      result.begin(), 
                      [global_sum_ptr = thrust::raw_pointer_cast(global_sum.data())] __device__ (int gs_i, int start_idx) {
                          return gs_i - global_sum_ptr[start_idx];
                      });

    return result;
}