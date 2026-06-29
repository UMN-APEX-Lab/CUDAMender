#include "kernel_launch.h"
#include "cuda_helpers.h"

int main() {
    launch(4, 256);
    cudaCheckErrors("kernel launch failed");
    launch(4, 16, 4, 16);
    cudaCheckErrors("kernel launch failed");
    launch(4, 16, 4, 16, 4, 1);
    cudaCheckErrors("kernel launch failed");
}