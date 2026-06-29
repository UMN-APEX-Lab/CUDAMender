#include "kernel_launch.h"
#include "kernel_helpers.h"
#include <cstring>
#include <nvtx3/nvToolsExt.h>

int main(int argc, char **argv) {
    bool perf = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--perf") == 0) perf = true;
    }

    if (!perf) {
        launch(4, 256);
        cudaCheckErrors("kernel launch failed");
        launch(4, 16, 4, 16);
        cudaCheckErrors("kernel launch failed");
        launch(4, 16, 4, 16, 4, 1);
        cudaCheckErrors("kernel launch failed");
    } else {
        const int WARMUP = 3;
        const int ITERS  = 100;

        for (int i = 0; i < WARMUP; i++) {
            launch(128, 256);
            cudaDeviceSynchronize();
        }

        nvtxRangePushA("bench_region");
        for (int i = 0; i < ITERS; i++) {
            launch(128, 256);
            cudaDeviceSynchronize();
        }
        nvtxRangePop();
    }
    return 0;
}