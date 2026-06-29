#include "square_indices.h"
#include <thrust/host_vector.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <nvtx3/nvtx3.hpp>

int main(int argc, char *argv[]) {
    bool perf = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--perf") == 0) perf = true;
    }

    if (!perf) {
        thrust::device_vector<int> x(5);

        square_indices(x);

        thrust::host_vector<int> expected {0, 1, 4, 9, 16};
        thrust::host_vector<int> h_x(x);

        for (std::size_t i = 0; i < x.size(); i++)
        {
            if (h_x[i] != expected[i])
            {
                std::cerr << "Error at index " << i << ": expected " << expected[i] << ", got "
                          << h_x[i] << std::endl;
                std::exit(1);
            }
        }
    } else {
        const std::size_t N = 50'000'000;
        thrust::device_vector<int> x(N);

        const int warmup_iters = 3;
        const int timed_iters  = 100;

        for (int i = 0; i < warmup_iters; ++i) {
            square_indices(x);
        }
        cudaDeviceSynchronize();

        nvtxRangePushA("bench_region");
        for (int i = 0; i < timed_iters; ++i) {
            square_indices(x);
        }
        nvtxRangePop();
        cudaDeviceSynchronize();
    }

    return 0;
}