#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * Bitonic sort is used here because it is highly efficient for small fixed-size 
 * arrays on GPUs. Since segmentSize is up to 128, we can implement a bitonic 
 * sort within the kernel.
 */

__global__ void k_sortSegments(float *array_d, float *arrayOut_d, int segmentSize, int arraySize) {
    // Each block handles multiple segments. 
    // Each segment is processed by a set of threads.
    // To handle arbitrary threads per block, we calculate the segment index based on blockIdx and threadIdx.
    
    int numSegments = arraySize / segmentSize;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // We assign one segment to a group of threads. 
    // However, for simplicity and to handle segmentSize up to 128, 
    // we can let each thread handle one element of one segment, 
    // and use a loop to process all segments assigned to this block.
    
    // To ensure we can sort up to 128 elements, we need at least 128 threads available 
    // or a way to map threads to elements.
    // Given the requirement "arbitrary number of threads per block", 
    // we'll use a strategy where we iterate through segments.
    
    // Calculate which segment this thread is working on.
    // We want to process segments in parallel.
    int segIdx = tid / 128; // Assuming we use 128 threads per segment for efficiency
    int localTid = tid % 128;

    if (segIdx < numSegments && localTid < segmentSize) {
        // Load segment into shared memory for fast access
        __shared__ float s_data[128 * 32]; // This is too large for shared mem if blockDim is large.
        // Instead, let's use a different approach: 
        // Process segments sequentially within a block or use global memory with synchronization.
    }
}

// Redefining the kernel to be more robust regarding blockDim and segmentSize.
__global__ void k_sortSegments_impl(float *array_d, float *arrayOut_d, int segmentSize, int arraySize) {
    int numSegments = arraySize / segmentSize;
    
    // Each block processes one or more segments.
    // Let's assume one block processes one segment for simplicity, 
    // but we must handle the case where blockDim.x != segmentSize.
    
    int segIdx = blockIdx.x;
    if (segIdx >= numSegments) return;

    float* segmentPtr = array_d + segIdx * segmentSize;
    float* outPtr = arrayOut_d + segIdx * segmentSize;

    // Use a local array in shared memory to perform the sort
    __shared__ float s_data[128];

    // Load data into shared memory
    for (int i = threadIdx.x; i < segmentSize; i += blockDim.x) {
        s_data[i] = segmentPtr[i];
    }
    __syncthreads();

    // Bitonic Sort
    for (int size = 2; size <= 128; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < segmentSize; i += blockDim.x) {
                int ixj = i ^ stride;
                if (ixj > i && ixj < segmentSize) {
                    float a = s_data[i];
                    float b = s_data[ixj];
                    
                    // Determine direction
                    bool ascending = ((i & size) == 0);
                    
                    if ((ascending && a > b) || (!ascending && a < b)) {
                        s_data[i] = b;
                        s_data[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write sorted data back to output
    for (int i = threadIdx.x; i < segmentSize; i += blockDim.x) {
        outPtr[i] = s_data[i];
    }
}

// To match the requested signature exactly:
__global__ void k_sortSegments(float *array_d, float *arrayOut_d, int segmentSize, int arraySize) {
    int numSegments = arraySize / segmentSize;
    int segIdx = blockIdx.x;
    
    if (segIdx >= numSegments) return;

    __shared__ float s_data[128];

    // Load
    for (int i = threadIdx.x; i < segmentSize; i += blockDim.x) {
        s_data[i] = array_d[segIdx * segmentSize + i];
    }
    __syncthreads();

    // Bitonic Sort
    // Note: Bitonic sort requires the size to be a power of 2. 
    // For non-powers of 2, we can treat the array as size 128 and fill with infinity.
    
    // Initialize padding with infinity
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        if (i >= segmentSize) s_data[i] = 1e38f; 
    }
    __syncthreads();

    for (int size = 2; size <= 128; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < 128; i += blockDim.x) {
                int ixj = i ^ stride;
                if (ixj > i) {
                    float a = s_data[i];
                    float b = s_data[ixj];
                    bool ascending = ((i & size) == 0);
                    if ((ascending && a > b) || (!ascending && a < b)) {
                        s_data[i] = b;
                        s_data[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Store (ignoring padding)
    for (int i = threadIdx.x; i < segmentSize; i += blockDim.x) {
        arrayOut_d[segIdx * segmentSize + i] = s_data[i];
    }
}