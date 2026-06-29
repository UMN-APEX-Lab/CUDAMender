#ifndef KERNEL_LAUNCH_H
#define KERNEL_LAUNCH_H

// Kernel function (implementation provided separately)
__global__ void kernel(int *output, const int *input);

// Function to implement: launch kernel with specified dimensions
void launch(int gridSizeX, int blockSizeX, int gridSizeY = 1, int blockSizeY = 1, 
            int gridSizeZ = 1, int blockSizeZ = 1);

#endif // KERNEL_LAUNCH_H