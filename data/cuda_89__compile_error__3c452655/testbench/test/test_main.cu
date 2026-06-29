#include "mix_and_scramble.h"
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

static void run_correctness_tests() {
    thrust::device_vector<int> x {1, 2, 3, 4, 5};
    thrust::device_vector<int> res(x.size());
    thrust::device_vector<int> ref {1, 2, 3, 4, 5};

    mix_and_scramble(x, res);

    if (res == ref)
    {
        std::cerr << "Error: result vector is the same before and after shuffle!" << std::endl;
        std::exit(1);
    }

    thrust::sort(res.begin(), res.end());

    if (res != ref)
    {
        std::cerr << "Error: elements are not the same as the original." << std::endl;
        std::exit(1);
    }
}

static void run_benchmark() {
    const int N = 50000000;
    const int warmup_iters = 3;
    const int timed_iters = 100;

    thrust::device_vector<int> x(N);
    thrust::sequence(x.begin(), x.end());
    thrust::device_vector<int> res(N);

    for (int i = 0; i < warmup_iters; i++) {
        mix_and_scramble(x, res);
    }
    cudaDeviceSynchronize();

    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed_iters; i++) {
        mix_and_scramble(x, res);
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

    run_correctness_tests();

    if (perf) {
        run_benchmark();
    }

    return 0;
}