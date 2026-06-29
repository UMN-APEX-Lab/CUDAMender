#include "has_negatives.h"
#include <thrust/device_vector.h>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

int main(int argc, char* argv[]) {
    bool perf = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--perf") == 0) perf = true;
    }

    auto dynamic_test = []() {
        thrust::device_vector<int> x {42, 0, 4, -2, 1};

        auto has = has_negatives(x);

        if (has != true)
        {
            std::cerr << "Error: expected true, got " << has << std::endl;
            std::exit(1);
        }

        x   = {42, 0, 4, 2, 1};
        has = has_negatives(x);

        if (has != false)
        {
            std::cerr << "Error: expected false, got " << has << std::endl;
            std::exit(1);
        }
    };

    dynamic_test();

    if (perf) {
        const int N = 50000000;
        const int warmup = 3;
        const int timed = 500;

        thrust::device_vector<int> large(N, 1);
        large[N - 1] = -1;

        for (int i = 0; i < warmup; ++i) {
            volatile bool r = has_negatives(large);
            (void)r;
        }
        cudaDeviceSynchronize();

        nvtxRangePushA("bench_region");
        for (int i = 0; i < timed; ++i) {
            volatile bool r = has_negatives(large);
            (void)r;
        }
        cudaDeviceSynchronize();
        nvtxRangePop();
    }

    return 0;
}