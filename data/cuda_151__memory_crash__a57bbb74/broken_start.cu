#include "miller_rabin.h"

__device__ unsigned long long power_mod(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) {
            __uint128_t temp = (__uint128_t)res * base;
            res = (unsigned long long)(temp % mod);
        }
        __uint128_t temp = (__uint128_t)base * base;
        base = (unsigned long long)(temp % mod);
        exp /= 2;
    }
    return res;
}

__global__ void k_millerRabin(unsigned long long *inputNumbers, unsigned long long *primalityResults, int numberOfElements) {
    // Dynamic shared memory for decomposition values: d and r
    // Each warp needs its own d and r.
    // We can use a simple array in shared memory. 
    // Since each warp needs 2 values (d and r), we allocate based on the number of warps in the block.
    unsigned long long* smem = (unsigned long long*)__cvta_generic_to_shared(0); 
    // Note: The prompt mentions dynamicSMemSizeMillerRabin, but since we are implementing the kernel, 
    // we access the shared memory allocated at launch.
    // Layout: [warp0_d, warp0_r, warp1_d, warp1_r, ...]
    
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpsPerBlock = blockDim.x / 32;
    int gridWarps = gridDim.x * warpsPerBlock;

    // Offset for this warp in shared memory
    int smemOffset = warpId * 2;

    for (int i = warpId + (blockIdx.x * warpsPerBlock) * 32; i < numberOfElements; i += gridWarps) {
        unsigned long long n = inputNumbers[i];
        
        // Handle small cases
        if (n < 2) {
            if (laneId == 0) primalityResults[i] = 0;
            __syncwarp();
            continue;
        }
        if (n == 2 || n == 3) {
            if (laneId == 0) primalityResults[i] = n;
            __syncwarp();
            continue;
        }
        if (n % 2 == 0) {
            if (laneId == 0) primalityResults[i] = 0;
            __syncwarp();
            continue;
        }

        // Decompose n-1 = d * 2^r
        if (laneId == 0) {
            unsigned long long d = n - 1;
            unsigned int r = 0;
            while (d % 2 == 0) {
                d /= 2;
                r++;
            }
            smem[smemOffset] = d;
            smem[smemOffset + 1] = r;
        }
        __syncwarp();

        unsigned long long d = smem[smemOffset];
        unsigned int r = (unsigned int)smem[smemOffset + 1];
        unsigned long long a = 2 + laneId;
        
        bool isComposite = false;
        if (a < n) {
            unsigned long long x = power_mod(a, d, n);
            if (x == 1 || x == n - 1) {
                // Probably prime for this witness
            } else {
                bool found = false;
                for (unsigned int j = 0; j < r - 1; j++) {
                    __uint128_t temp = (__uint128_t)x * x;
                    x = (unsigned long long)(temp % n);
                    if (x == n - 1) {
                        found = true;
                        break;
                    }
                }
                if (!found) isComposite = true;
            }
        } else {
            // Witness a >= n, doesn't provide info, treat as not composite
            isComposite = false;
        }

        // Aggregate results across the warp
        // Use __ballot_sync to see if any thread found the number to be composite
        unsigned int mask = __ballot_sync(0xFFFFFFFF, isComposite);
        
        if (laneId == 0) {
            if (mask == 0) {
                primalityResults[i] = n;
            } else {
                primalityResults[i] = 0;
            }
        }
        __syncwarp();
    }
}