#include "local_prefix_sum.h"
#include <thrust/host_vector.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

void run_correctness_test() {
    thrust::device_vector<int> vec {1, 2, 3, 2, 2, 3, 3, 5, 3, 2};
    thrust::device_vector<int> keys {0, 0, 0, 1, 1, 1, 1, 2, 3, 3};

    auto res = local_prefix_exclusive_sum(vec, keys);
    thrust::host_vector<int> res_host(res);

    thrust::host_vector<int> expected {0, 1, 3, 0, 2, 4, 7, 0, 0, 3};

    for (size_t i = 0; i < res_host.size(); i++)
    {
        if (res_host[i] != expected[i])
        {
            std::cerr << "Error, expected " << expected[i] << " but got " << res_host[i]
                      << " at index " << i << std::endl;
            std::exit(1);
        }
    }
}

void run_benchmark() {
    const int N = 10'000'000;
    const int num_segments = 50'000;
    const int seg_len = N / num_segments;

    thrust::host_vector<int> h_vec(N);
    thrust::host_vector<int> h_keys(N);
    for (int i = 0; i < N; i++) {
        h_vec[i] = (i % 7) + 1;
        h_keys[i] = i / seg_len;
    }

    thrust::device_vector<int> d_vec(h_vec);
    thrust::device_vector<int> d_keys(h_keys);

    const int warmup_iters = 3;
    const int timed_iters = 100;

    for (int i = 0; i < warmup_iters; i++) {
        auto res = local_prefix_exclusive_sum(d_vec, d_keys);
    }
    cudaDeviceSynchronize();

    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed_iters; i++) {
        auto res = local_prefix_exclusive_sum(d_vec, d_keys);
    }
    cudaDeviceSynchronize();
    nvtxRangePop();
}

int main(int argc, char* argv[]) {
    bool perf = false;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--perf") == 0) {
            perf = true;
        }
    }

    if (perf) {
        run_benchmark();
    } else {
        run_correctness_test();
    }
    return 0;
}