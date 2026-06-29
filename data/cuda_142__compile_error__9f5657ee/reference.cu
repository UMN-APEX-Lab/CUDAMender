#include "nbody_kernel.h"

// This kernel calculates the forces acting on each particle independently, utilizing shared-memory tiling to optimize the accesses by each thread.
__global__ void k_calculateNBodyForce(float *xPosition_d, float *yPosition_d, float *mass_d, float *xForce_d, float *yForce_d) {
    extern __shared__ float s_tile[];
    constexpr int Y_POS_OFFSET = NUM_PARTICLES_PER_TILE;
    constexpr int MASS_OFFSET = Y_POS_OFFSET + NUM_PARTICLES_PER_TILE;
    constexpr int TILE_END = MASS_OFFSET + NUM_PARTICLES_PER_TILE;

    // Order of the input properties of particle. x = 0, y = 1, mass = 2
    constexpr int X_PROPERTY_INDEX = 0;
    constexpr int Y_PROPERTY_INDEX = 1;
    constexpr int MASS_PROPERTY_INDEX = 2;

    // Number of tiles to calculate per block.
    constexpr int NUM_TILE_ITERATIONS = 1 + (NUM_PARTICLES - 1) / NUM_PARTICLES_PER_TILE;

    // The grid-stride loop allows for a flexible number of particles to be used in the initial particle of the all versus all force calculation.
    int numGridThreads = blockDim.x * gridDim.x;
    int numGridStrideLoopIterations = 1 + (NUM_PARTICLES - 1) / numGridThreads;
    int threadIndexInGrid = threadIdx.x + blockIdx.x * blockDim.x;

    // Block-stride for tiles to load all three components (x, y, mass) of particles together, enabling the use of more threads than the tile size.
    int numBlockThreads = blockDim.x;
    int numBlockStrideLoopIterations = 1 + (NUM_PARTICLES_PER_TILE * NUMBER_OF_INPUT_PROPERTIES_OF_PARTICLE - 1) / numBlockThreads;
    int threadIndexInBlock = threadIdx.x;

    for(int gridIteration = 0; gridIteration < numGridStrideLoopIterations; gridIteration++) {
        int itemId = gridIteration * numGridThreads + threadIndexInGrid;
        float xPositionCurrent, yPositionCurrent, massCurrent;

        if(itemId < NUM_PARTICLES) {
            xPositionCurrent = xPosition_d[itemId];
            yPositionCurrent = yPosition_d[itemId];
            massCurrent = mass_d[itemId];
        }

        float xForce = 0.0f;
        float yForce = 0.0f;

        // Iterating for all tiles to compute the forces acting on them.
        for(int tileIteration = 0; tileIteration < NUM_TILE_ITERATIONS; tileIteration++) {
            int currentTileOffset = tileIteration * NUM_PARTICLES_PER_TILE;

            // All block threads exit the loop when work is complete.
            if(currentTileOffset >= NUM_PARTICLES) {
                break;
            }

            // Loading tile particles into shared memory. 3x threads can load 1x particles by working on x, y, mass components together to decrease the number of block-stride iterations.
            // The access pattern is coalesced, and each global element is fetched a single time.
            for(int blockIteration = 0; blockIteration < numBlockStrideLoopIterations; blockIteration++) {
                int tileItemId = blockIteration * numBlockThreads + threadIndexInBlock;

                if(tileItemId < Y_POS_OFFSET) {
                    // Loading x position of particle.
                    s_tile[tileItemId] =    (currentTileOffset + tileItemId - NUM_PARTICLES_PER_TILE * X_PROPERTY_INDEX < NUM_PARTICLES) ?
                                            xPosition_d[currentTileOffset + tileItemId - NUM_PARTICLES_PER_TILE * X_PROPERTY_INDEX] :
                                            0.0f;
                } else if(tileItemId < MASS_OFFSET) {
                    // Loading y position of particle.
                    s_tile[tileItemId] =    (currentTileOffset + tileItemId - NUM_PARTICLES_PER_TILE * Y_PROPERTY_INDEX < NUM_PARTICLES) ?
                                            yPosition_d[currentTileOffset + tileItemId - NUM_PARTICLES_PER_TILE * Y_PROPERTY_INDEX] :
                                            0.0f;
                } else if(tileItemId < TILE_END) {
                    // Loading mass of particle.
                    s_tile[tileItemId] =    (currentTileOffset + tileItemId - NUM_PARTICLES_PER_TILE * MASS_PROPERTY_INDEX < NUM_PARTICLES) ?
                                            mass_d[currentTileOffset + tileItemId - NUM_PARTICLES_PER_TILE * MASS_PROPERTY_INDEX] :
                                            0.0f;
                }
            }

            // Waiting for all block threads to finish loading the current tile.
            __syncthreads();

            // Iterating over the tile elements and computing a force for each particle in the tile relative to the current particle. Shared memory has a broadcasting ability, so all threads accessing the same element incur no extra cost.
            for(int tileItem = 0; tileItem < NUM_PARTICLES_PER_TILE; tileItem++) {
                float xPosition = s_tile[tileItem + NUM_PARTICLES_PER_TILE * X_PROPERTY_INDEX];
                float yPosition = s_tile[tileItem + NUM_PARTICLES_PER_TILE * Y_PROPERTY_INDEX];
                float mass =      s_tile[tileItem + NUM_PARTICLES_PER_TILE * MASS_PROPERTY_INDEX];
                float dx = xPositionCurrent - xPosition;
                float dy = yPositionCurrent - yPosition;
                float massMultiplied = massCurrent * mass;
                float distanceSquared = dx * dx + dy * dy;
                float inverseDistance = 1.0f / sqrtf(distanceSquared + SMOOTHING);

                // Zeroing the force when two particles occupy the same position or when two particles are identical.
                float forceMultiplier = (distanceSquared < VERY_CLOSE) ? 
                                        0.0f : 
                                        -GRAVITATIONAL_FORCE_CONSTANT * massMultiplied * 
                                        inverseDistance * inverseDistance * inverseDistance;
                xForce = fmaf(dx, forceMultiplier, xForce);
                yForce = fmaf(dy, forceMultiplier, yForce);
            }

            // Waiting all block threads to finish computing before loading next tile.
            __syncthreads();
        }

        if(itemId < NUM_PARTICLES) {
            xForce_d[itemId] = xForce;
            yForce_d[itemId] = yForce;
        }
    }
}