#include "include/radix_sort.h"
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>

namespace cg = cooperative_groups;

__global__ void k_radixSort(unsigned int* keysIn_d, unsigned int* keysOut_d, 
                            unsigned int* globalHistograms_d, int numElements) {
    using BlockLoad = cub::BlockLoad<unsigned int, BLOCK_DIM_X, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<unsigned int, BLOCK_DIM_X, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockScan = cub::BlockScan<unsigned int, BLOCK_DIM_X, cub::BLOCK_SCAN_WARP_SCANS>;
    
    extern __shared__ unsigned int temp_storage[];
    
    unsigned int* block_histogram = (unsigned int*)temp_storage;
    unsigned int* block_scan = block_histogram + RADIX_SIZE * BLOCK_DIM_X;
    unsigned int* thread_offsets = block_scan + BLOCK_DIM_X;
    
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane_id = threadIdx.x & 31;
    unsigned int thread_id = threadIdx.x;
    
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    
    unsigned int local_keys[ITEMS_PER_THREAD];
    unsigned int local_digits[ITEMS_PER_THREAD];
    unsigned int local_offsets[ITEMS_PER_THREAD];
    unsigned int local_scan = 0;
    
    // Process all 8 passes for 32-bit integers with 4-bit radix
    for (int pass = 0; pass < NUM_PASSES; pass++) {
        int shift = pass * RADIX_BITS;
        
        // Phase 1: Load keys and extract digits
        BlockLoad(block, keysIn_d + tid, local_keys, numElements - tid, 0u);
        
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            local_digits[i] = (local_keys[i] >> shift) & (RADIX_SIZE - 1);
        }
        
        // Phase 2: Compute block-level histogram
        #pragma unroll
        for (int r = 0; r < RADIX_SIZE; r++) {
            block_histogram[r * BLOCK_DIM_X + threadIdx.x] = 0;
        }
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            unsigned int digit = local_digits[i];
            if (local_keys[i] != 0u || tid + i < numElements) {
                atomicAdd(&block_histogram[digit * BLOCK_DIM_X + threadIdx.x], 1u);
            }
        }
        __syncthreads();
        
        // Reduce histogram across threads in block
        unsigned int block_hist[RADIX_SIZE] = {0};
        #pragma unroll
        for (int r = 0; r < RADIX_SIZE; r++) {
            unsigned int sum = 0;
            #pragma unroll
            for (int t = 0; t < BLOCK_DIM_X; t++) {
                sum += block_histogram[r * BLOCK_DIM_X + t];
            }
            block_hist[r] = sum;
        }
        
        // Write block histogram to global memory
        if (threadIdx.x == 0) {
            unsigned int* block_hist_ptr = globalHistograms_d + pass * RADIX_SIZE * gridDim.x + blockIdx.x * RADIX_SIZE;
            #pragma unroll
            for (int r = 0; r < RADIX_SIZE; r++) {
                block_hist_ptr[r] = block_hist[r];
            }
        }
        __syncthreads();
        
        // Grid-wide synchronization before prefix sum
        if (threadIdx.x == 0) {
            atomicAdd(&globalHistograms_d[pass * RADIX_SIZE * gridDim.x + gridDim.x * RADIX_SIZE + blockIdx.x], 1u);
        }
        grid.sync();
        
        // Phase 3: Compute global prefix sum offsets
        // Each block reads all block histograms and computes prefix sums
        unsigned int* global_hist_base = globalHistograms_d + pass * RADIX_SIZE * gridDim.x;
        unsigned int prefix_sum[RADIX_SIZE] = {0};
        
        if (threadIdx.x == 0) {
            for (int r = 0; r < RADIX_SIZE; r++) {
                unsigned int sum = 0;
                for (int b = 0; b < blockIdx.x; b++) {
                    sum += global_hist_base[b * RADIX_SIZE + r];
                }
                prefix_sum[r] = sum;
            }
        }
        __syncthreads();
        
        // Broadcast prefix sums to all threads
        #pragma unroll
        for (int r = 0; r < RADIX_SIZE; r++) {
            thread_offsets[r] = prefix_sum[r];
        }
        
        // Phase 4: Compute per-thread offsets within block
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            unsigned int digit = local_digits[i];
            local_scan = 0;
            #pragma unroll
            for (int t = 0; t < threadIdx.x; t++) {
                if (local_digits[t] == digit) {
                    local_scan++;
                }
            }
            local_offsets[i] = thread_offsets[digit] + local_scan;
        }
        
        __syncthreads();
        
        // Phase 5: Scatter keys to output
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            if (local_keys[i] != 0u || tid + i < numElements) {
                keysOut_d[local_offsets[i]] = local_keys[i];
            }
        }
        
        // Grid-wide synchronization before next pass
        grid.sync();
        
        // Swap input/output pointers for next pass
        unsigned int* temp = keysIn_d;
        keysIn_d = keysOut_d;
        keysOut_d = temp;
    }
}