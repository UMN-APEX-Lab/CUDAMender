#include "negate_and_prefix_sum.h"
#include <thrust/device_vector.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

int main(int argc, char* argv[]) {
    bool perf_mode = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--perf") == 0) {
            perf_mode = true;
        }
    }

    auto dynamic_test = []() {
        thrust::device_vector<int> vec {1, -3, 2, -4, 5};
        auto res = negate_and_prefix_sum(vec);
        thrust::device_vector<int> expected {0, -1, 2, 0, 4};

        for (int i = 0; i < res.size(); i++)
        {
            if (res[i] != expected[i])
            {
                std::cerr << "Error: error at index " << i << " got " << res[i] << " expected "
                          << expected[i] << std::endl;
                std::exit(1);
            }
        }
    };

    dynamic_test();

    if (perf_mode) {
        const int N = 10000000;
        thrust::device_vector<int> large_vec(N);
        for (int i = 0; i < N; i++) {
            large_vec[i] = (i % 201) - 100;
        }

        const int warmup_iters = 3;
        const int timed_iters = 100;

        for (int i = 0; i < warmup_iters; i++) {
            auto res = negate_and_prefix_sum(large_vec);
            cudaDeviceSynchronize();
        }

        nvtxRangePushA("bench_region");
        for (int i = 0; i < timed_iters; i++) {
            auto res = negate_and_prefix_sum(large_vec);
            cudaDeviceSynchronize();
        }
        nvtxRangePop();
    }

    return 0;
}