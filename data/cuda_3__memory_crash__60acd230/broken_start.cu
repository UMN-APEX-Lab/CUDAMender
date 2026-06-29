#include "kernel_launch.h"
#include <cuda_runtime.h>

void launch(int gridSizeX, int blockSizeX, int gridSizeY, int blockSizeY, 
            int gridSizeZ, int blockSizeZ) {
    // Define pointers as requested. 
    // The problem states we do not need to allocate memory or initialize these.
    int *d_output = nullptr;
    const int *d_input = nullptr;

    dim3 grid(gridSizeX, gridSizeY, gridSizeZ);
    dim3 block(blockSizeX, blockSizeY, blockSizeZ);

    // Using cudaLaunchKernel to avoid triple chevrons (<<< >>>)
    void* args[] = { &d_output, &d_input };
    
    cudaLaunchKernel(
        (const void*)kernel, 
        grid, 
        block, 
        args, 
        0, 
        nullptr
    );
}
