#ifndef NBODY_KERNEL_H
#define NBODY_KERNEL_H

// Algorithm-related constants
constexpr int NUM_PARTICLES = 5;
constexpr int ALLOCATION_SIZE = NUM_PARTICLES * sizeof(float);
constexpr int NUMBER_OF_INPUT_PROPERTIES_OF_PARTICLE = 3;
constexpr float GRAVITATIONAL_FORCE_CONSTANT = 10000.0f;
constexpr float VERY_CLOSE = 1e-15f;
constexpr float SMOOTHING = 1e-5f;

// CUDA-related constants
constexpr int NUM_PARTICLES_PER_TILE = 2;
constexpr int SHARED_MEM_ALLOCATION_SIZE = NUM_PARTICLES_PER_TILE * NUMBER_OF_INPUT_PROPERTIES_OF_PARTICLE * sizeof(float);

// Kernel declaration
__global__ void k_calculateNBodyForce(float *xPosition_d, float *yPosition_d, float *mass_d, float *xForce_d, float *yForce_d);

#endif // NBODY_KERNEL_H