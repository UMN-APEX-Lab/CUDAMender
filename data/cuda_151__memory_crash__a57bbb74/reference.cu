#include "miller_rabin.h"

__global__ void k_millerRabin(unsigned long long* inputNumbers, unsigned long long* primalityResults, int numberOfElements) {
    
    // Compute the number of warps per block.
    int numWarpsPerBlock = blockDim.x / warpSize;

    // Partition shared memory into two:
    // s_millerComponent for the Miller decomposition
    // s_exponentFactor for the exponent factor
    extern __shared__ char sharedBuffer[];
    unsigned long long* s_millerComponent = (unsigned long long*)sharedBuffer;
    int* s_exponentFactor = (int*)(sharedBuffer + sizeof(unsigned long long) * numWarpsPerBlock);

    int globThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = globThreadId / warpSize;
    int laneId = threadIdx.x % warpSize;

    // Local warp id within the block.
    int localWarpId = threadIdx.x / warpSize;

    // Total number of warps in a grid
    int numWarps = gridDim.x * blockDim.x / warpSize;
    
    // When the dataset is greater than the grid dim, kernel will break it into chunks and each warps processes some chunks of data
    // Warp 0 processes index 0, 4, 8 etc
    // Warp 1 processes index 1, 5, 9 etc
    for(int currentWarp = warpId; currentWarp < numberOfElements; currentWarp += numWarps) {
        unsigned long long num = inputNumbers[currentWarp];
        
        // Initializing flag to assume the number is prime
        int isPrime = 1;

        // Ensures only one thread per warp performs the decomposition
        if (laneId == 0) {

            // Decompose of input numbers num-1 
            unsigned long long millerComponent = num - 1;
            int exponentFactor = 0;
            
            // Factoring out powers of 2
            while ((millerComponent & 1) == 0) {
                millerComponent >>= 1;
                exponentFactor++;
            }
            s_millerComponent[localWarpId] = millerComponent;
            s_exponentFactor[localWarpId] = exponentFactor;
        }

        // Synchronize warp
        __syncwarp();

        unsigned long long millerComponent = s_millerComponent[localWarpId];
        int exponentFactor = s_exponentFactor[localWarpId];

        // Exclude numbers less than 2
        isPrime *= (num >= 2);

        // Checks for small primes
        isPrime *= (num == 2 || num == 3 || num % 2 != 0);

        // Choosing base 'a'  
        unsigned long long a = 2 + (laneId % (num - 4 + (num <= 4)));

        // Modular exponent setup
        unsigned long long x = 1, base = a, power = millerComponent;
        
        // Performming modular exponentiation
        while (power > 0) {
            if (power & 1) x = (x * base) % num;
            base = (base * base) % num;
            power >>= 1;
        }

        // Initial check for non-trivial roots
        int localPrime = (x == 1 || x == num - 1);
        
        // Performing squaring to check for composite witnesses
        #pragma unroll
        for (int r = 1; r < exponentFactor && !localPrime; r++) {
            x = (x * x) % num;
            if (x == num - 1) localPrime = 1;
        }

        // Updating primality based witness 
        isPrime *= localPrime;

        // Retrieve the active lane mask for the current warp, which is used by __shfl_down_sync to combine 
        // the 'isPrime' flags across all active threads in the warp.
        unsigned int mask = __activemask();
        int allPrime = isPrime;

        // Communication across threads to ensure primality results within warp.
        #pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            allPrime &= __shfl_down_sync(mask, allPrime, offset);
        }

        // Ensures only one thread per warp writes the result
        if (laneId == 0) {
            // Store result in output array 
            primalityResults[currentWarp] = allPrime ? num : 0;
        }
    }
}