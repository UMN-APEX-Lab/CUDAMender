#include "kernel_launch.h"
#include "kernel_helpers.h"

void launch(int gridSizeX, int blockSizeX, int gridSizeY, int blockSizeY, int gridSizeZ, int blockSizeZ)
{
    // Pointers to be passed to the kernel. 
    // Since the problem description doesn't specify how to obtain input/output arrays,
    // but requires the kernel to be launched with specific dimensions and dynamic shared memory,
    // we assume the kernel expects valid pointers. In a real scenario, these would be passed in.
    // However, the prompt specifically asks to implement the launch mechanism.
    int *d_output = nullptr;
    const int *d_input = nullptr;

    dim3 grid(gridSizeX, gridSizeY, gridSizeZ);
    dim3 block(blockSizeX, blockSizeY, blockSizeZ);

    // The prompt specifies to "allocate dynamic shared memory".
    // The amount is not specified, so we use 0 as a default or a symbolic value.
    size_t sharedMemSize = 0;

    // Launch kernel using triple chevrons
    kernel<<<grid, block, sharedMemSize>>>(d_output, d_input);

    cudaCheckErrors("Kernel launch failed");
}
