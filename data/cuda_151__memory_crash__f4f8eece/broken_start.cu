#include "miller_rabin.h"
#include <cuda_runtime.h>

// Helper function to compute modular exponentiation: (base^exp) % mod
__device__ unsigned long long modPow(unsigned long long base, unsigned long long exp, unsigned long long mod) {
    unsigned long long result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp & 1) {
            result = (__uint128_t)result * base % mod;
        }
        exp = exp >> 1;
        base = (__uint128_t)base * base % mod;
    }
    return result;
}

__global__ void k_millerRabin(unsigned long long *inputNumbers, unsigned long long *primalityResults, int numberOfElements) {
    extern __shared__ unsigned char sharedMem[];
    
    // Shared memory layout per warp: d (8 bytes), r (4 bytes), isComposite flag (4 bytes), maybePrime flag (4 bytes)
    // We need 20 bytes per warp, but let's use 32 bytes for alignment
    const int bytesPerWarp = 32;
    int warpsPerBlock = (blockDim.x + 31) / 32;
    int warpIdInBlock = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    
    // Pointer to this warp's shared memory
    unsigned char* warpSharedMem = sharedMem + warpIdInBlock * bytesPerWarp;
    unsigned long long* sharedD = (unsigned long long*)(warpSharedMem);
    unsigned int* sharedR = (unsigned int*)(warpSharedMem + 8);
    unsigned int* sharedIsComposite = (unsigned int*)(warpSharedMem + 12);
    unsigned int* sharedMaybePrime = (unsigned int*)(warpSharedMem + 16);
    
    int totalWarps = gridDim.x * warpsPerBlock;
    int globalWarpId = blockIdx.x * warpsPerBlock + warpIdInBlock;
    
    // Strided loop over numbers
    for (int idx = globalWarpId; idx < numberOfElements; idx += totalWarps) {
        unsigned long long n = inputNumbers[idx];
        
        // Handle edge cases
        if (n < 2) {
            if (laneId == 0) {
                primalityResults[idx] = 0;
            }
            continue;
        }
        if (n == 2 || n == 3) {
            if (laneId == 0) {
                primalityResults[idx] = n;
            }
            continue;
        }
        if (n % 2 == 0) {
            if (laneId == 0) {
                primalityResults[idx] = 0;
            }
            continue;
        }
        
        // First thread in warp computes decomposition n-1 = d * 2^r
        if (laneId == 0) {
            unsigned long long d = n - 1;
            unsigned int r = 0;
            while ((d & 1) == 0) {
                d >>= 1;
                r++;
            }
            *sharedD = d;
            *sharedR = r;
            *sharedIsComposite = 0;
            *sharedMaybePrime = 0;
        }
        __syncwarp();
        
        unsigned long long d = *sharedD;
        unsigned int r = *sharedR;
        
        // Each thread tests a different witness base
        // Witness base is 2 + laneId (so bases 2, 3, 4, 5, ...)
        unsigned long long a = 2 + laneId;
        
        // Skip if base >= n-1 (only relevant for small n)
        bool isComposite = false;
        if (a < n - 1) {
            unsigned long long x = modPow(a, d, n);
            
            if (x != 1 && x != n - 1) {
                bool continueLoop = true;
                for (unsigned int i = 1; i < r && continueLoop; i++) {
                    x = (__uint128_t)x * x % n;
                    if (x == n - 1) {
                        continueLoop = false;
                    }
                }
                if (continueLoop) {
                    // x never became n-1, so composite
                    isComposite = true;
                }
            }
        }
        
        // Warp-level reduction: if any thread found composite, mark as composite
        // Use ballot to check if any thread found composite
        unsigned int compositeBallot = __ballot_sync(0xFFFFFFFF, isComposite);
        
        if (laneId == 0) {
            if (compositeBallot != 0) {
                *sharedIsComposite = 1;
                primalityResults[idx] = 0;
            } else {
                // All witnesses passed, number is probably prime
                primalityResults[idx] = n;
            }
        }
        __syncwarp();
    }
}