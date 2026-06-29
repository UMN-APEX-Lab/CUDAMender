#include "radix_sort.h"

__global__ void k_radixSort(unsigned int *keysIn_d,
                            unsigned int *keysOut_d,
                            unsigned int *globalHistograms_d,
                            int numElements) {

    // NOTE: This kernel uses cooperative groups for grid-wide synchronization.
    // Tesla T4 limits cooperative kernels to ~128-256 blocks, restricting us to
    // ~131K-262K elements max. For 1M+ elements, use non-cooperative approach.

    // Device-only constant
    constexpr int BITS_PER_INTEGER = 32;

    // CUB type definitions
    typedef cub::BlockLoad<unsigned int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    typedef cub::BlockStore<unsigned int, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
    typedef cub::BlockScan<int, BLOCK_SIZE> BlockScanT;

    // Union for memory-efficient shared memory usage
    __shared__ union {
        typename BlockLoadT::TempStorage load;
        typename BlockStoreT::TempStorage store;
        typename BlockScanT::TempStorage scan;
    } s_tempStorage;

    // Get cooperative grid group for synchronization
    cg::grid_group grid = cg::this_grid();

    int threadId = threadIdx.x;
    int blockId = blockIdx.x;
    int blockOffset = blockId * BLOCK_SIZE * ITEMS_PER_THREAD;
    int elementsInBlock = min(BLOCK_SIZE * ITEMS_PER_THREAD, numElements - blockOffset);

    // Shared memory for operations
    __shared__ int s_blockHistogram[RADIX_SIZE];
    __shared__ int s_globalOffsets[RADIX_SIZE];
    __shared__ int s_blockStarts[RADIX_SIZE];

    // Thread data storage
    unsigned int threadKeys[ITEMS_PER_THREAD];
    unsigned int threadDigits[ITEMS_PER_THREAD];
    int threadRanks[ITEMS_PER_THREAD];

    // Input/output buffer pointers
    unsigned int *src = keysIn_d;
    unsigned int *dst = keysOut_d;

    // Process each bit group (RADIX_PASSES passes for BITS_PER_INTEGER-bit integers, RADIX_BITS
    // bits per pass)
    for(int bitShift = 0; bitShift < BITS_PER_INTEGER; bitShift += RADIX_BITS) {
        
        // PHASE 1: Load data and count histogram
        // Time: O(n/p) per block, Space: O(r) shared memory
        // Each thread processes ITEMS_PER_THREAD elements, coalesced loads take O(n/p) time per block
        if(threadId < RADIX_SIZE) {
            s_blockHistogram[threadId] = 0;
        }
        __syncthreads();

        // Load data using CUB BlockLoad
        BlockLoadT(s_tempStorage.load).Load(src + blockOffset, threadKeys, elementsInBlock, 0);
        __syncthreads();

        // Extract digits and count in histogram
        for(int i = 0; i < ITEMS_PER_THREAD; i++) {
            if(threadId * ITEMS_PER_THREAD + i < elementsInBlock) {
                threadDigits[i] = (threadKeys[i] >> bitShift) & RADIX_MASK;
                atomicAdd(&s_blockHistogram[threadDigits[i]], 1);
            } else {
                threadDigits[i] = 0;
                threadKeys[i] = 0;
            }
        }
        __syncthreads();

        // Store block histogram to global memory
        if(threadId < RADIX_SIZE) {
            globalHistograms_d[blockId * RADIX_SIZE + threadId] = s_blockHistogram[threadId];
        }

        // Grid sync to ensure all histograms are computed
        grid.sync();

        // PHASE 2: Compute global prefix sums (only in block 0)
        // Time: O(r * B) single-threaded, Space: O(r) shared + O(r * B) global
        // Sequential scan over r radix buckets across B blocks, dominated by O(r * B) term
        if(blockId == 0) {
            if(threadId < RADIX_SIZE) {
                // Sum across all blocks
                int total = 0;
                for(int b = 0; b < gridDim.x; b++) {
                    total += globalHistograms_d[b * RADIX_SIZE + threadId];
                }
                s_blockHistogram[threadId] = total;
            }
            __syncthreads();

            // Compute global prefix scan and store in GLOBAL memory
            if(threadId == 0) {
                // Use the end of globalHistograms_d array to store global offsets
                unsigned int *globalOffsetsPtr = globalHistograms_d + gridDim.x * RADIX_SIZE;

                globalOffsetsPtr[0] = 0;
                for(int i = 1; i < RADIX_SIZE; i++) {
                    globalOffsetsPtr[i] = globalOffsetsPtr[i - 1] + s_blockHistogram[i - 1];
                }
            }
            __syncthreads();
        }

        // Grid sync to ensure offsets are computed
        grid.sync();

        // PHASE 3: Read global offsets and compute block starting positions
        // Time: O(r * B) per block, Space: O(r) shared memory
        // Each block computes starting positions by summing over previous blocks for each radix
        unsigned int *globalOffsetsPtr = globalHistograms_d + gridDim.x * RADIX_SIZE;

        if(threadId < RADIX_SIZE) {
            s_globalOffsets[threadId] = globalOffsetsPtr[threadId];
        }
        __syncthreads();

        // Compute block starting positions
        if(threadId < RADIX_SIZE) {
            int startPos = s_globalOffsets[threadId];
            // Add contributions from previous blocks
            for(int b = 0; b < blockId; b++) {
                startPos += globalHistograms_d[b * RADIX_SIZE + threadId];
            }
            s_blockStarts[threadId] = startPos;
        }
        __syncthreads();

        // PHASE 4: Compute local ranks using CUB BlockScan for each digit
        // Time: O(r * n/p) per block, Space: O(ITEMS_PER_THREAD) per thread + CUB temp storage
        // Performs r separate block scans, each taking O(n/p) time, for complete local ranking
        for(int i = 0; i < ITEMS_PER_THREAD; i++) {
            threadRanks[i] = -1;
        }

        for(int digit = 0; digit < RADIX_SIZE; digit++) {
            // Create binary array for this digit
            int threadFlags[ITEMS_PER_THREAD];
            for(int i = 0; i < ITEMS_PER_THREAD; i++) {
                if(threadId * ITEMS_PER_THREAD + i < elementsInBlock && threadDigits[i] == digit) {
                    threadFlags[i] = 1;
                } else {
                    threadFlags[i] = 0;
                }
            }

            // Compute exclusive scan to get ranks
            int scannedFlags[ITEMS_PER_THREAD];
            BlockScanT(s_tempStorage.scan).ExclusiveSum(threadFlags, scannedFlags);
            __syncthreads();

            // Store ranks for elements with this digit
            for(int i = 0; i < ITEMS_PER_THREAD; i++) {
                if(threadId * ITEMS_PER_THREAD + i < elementsInBlock && threadDigits[i] == digit &&
                   threadFlags[i] == 1) {
                    threadRanks[i] = scannedFlags[i];
                }
            }
        }

        // PHASE 5: Scatter elements to final positions
        // Time: O(n/p) per block, Space: O(ITEMS_PER_THREAD) per thread
        // Each thread scatters its ITEMS_PER_THREAD elements using computed ranks and offsets
        for(int i = 0; i < ITEMS_PER_THREAD; i++) {
            if(threadId * ITEMS_PER_THREAD + i < elementsInBlock && threadRanks[i] >= 0) {
                unsigned int digit = threadDigits[i];
                int finalPos = s_blockStarts[digit] + threadRanks[i];

                if(finalPos >= 0 && finalPos < numElements) {
                    dst[finalPos] = threadKeys[i];
                }
            }
        }

        // Grid sync before buffer swap
        grid.sync();

        // Swap source and destination for next iteration
        unsigned int *temp = src;
        src = dst;
        dst = temp;
    }
}