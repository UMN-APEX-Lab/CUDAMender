#include "nbody_kernel.h"
#include <cuda_runtime.h>

__global__ void k_calculateNBodyForce(float *xPosition_d, float *yPosition_d, float *mass_d, float *xForce_d, float *yForce_d) {
    // Shared memory for tiling particle data: x, y, and mass
    __shared__ float tileData[NUM_PARTICLES_PER_TILE][NUMBER_OF_INPUT_PROPERTIES_OF_PARTICLE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    float fx = 0.0f;
    float fy = 0.0f;
    float my_x = 0.0f;
    float my_y = 0.0f;
    float my_m = 0.0f;

    // Grid-stride loop to handle cases where NUM_PARTICLES > total threads
    for (int idx = i; idx < NUM_PARTICLES; idx += blockDim.x * gridDim.x) {
        my_x = xPosition_d[idx];
        my_y = yPosition_d[idx];
        my_m = mass_d[idx];

        // Iterate over particles in tiles
        for (int tile = 0; tile < (NUM_PARTICLES + NUM_PARTICLES_PER_TILE - 1) / NUM_PARTICLES_PER_TILE; ++tile) {
            // Cooperatively load tile into shared memory
            int loadIdx = tile * NUM_PARTICLES_PER_TILE + threadIdx.x;
            if (loadIdx < NUM_PARTICLES) {
                tileData[threadIdx.x][0] = xPosition_d[loadIdx];
                tileData[threadIdx.x][1] = yPosition_d[loadIdx];
                tileData[threadIdx.x][2] = mass_d[loadIdx];
            } else {
                tileData[threadIdx.x][0] = 0.0f;
                tileData[threadIdx.x][1] = 0.0f;
                tileData[threadIdx.x][2] = 0.0f;
            }

            __syncthreads();

            // Calculate force contribution from particles in the current tile
            for (int j = 0; j < NUM_PARTICLES_PER_TILE; ++j) {
                int globalJ = tile * NUM_PARTICLES_PER_TILE + j;
                if (globalJ < NUM_PARTICLES) {
                    float dx = tileData[j][0] - my_x;
                    float dy = tileData[j][1] - my_y;
                    float distSq = dx * dx + dy * dy;

                    // Avoid self-interaction or extremely close particles
                    if (distSq >= VERY_CLOSE) {
                        // Apply smoothing for numerical stability
                        float distSqSmoothed = distSq + SMOOTHING * SMOOTHING;
                        float distInv = 1.0f / sqrtf(distSqSmoothed);
                        float distInv3 = distInv * distInv * distInv;
                        
                        float forceMag = GRAVITATIONAL_FORCE_CONSTANT * my_m * tileData[j][2] * distInv3;
                        
                        fx += forceMag * dx;
                        fy += forceMag * dy;
                    }
                }
            }
            __syncthreads();
        }
        
        xForce_d[idx] = fx;
        yForce_d[idx] = fy;
    }
}
