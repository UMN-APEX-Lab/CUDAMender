#define _USE_MATH_DEFINES

#include "predicate_filter.h"
#include <cuda_runtime_api.h>
#include <thrust/partition.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/execution_policy.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <random>
#include <cstring>
#undef NDEBUG
#include <assert.h>

#define TEST_CASES 10
#define RAND_SEED 571149
#define UNIFORM_DIST_LOWER_BOUND -20
#define UNIFORM_DIST_UPPER_BOUND 10

#define CUDA_CHECK(call)                                        \
do {                                                            \
        cudaError_t error = call;                               \
        if (error != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",       \
                    __FILE__, __LINE__,                         \
                    cudaGetErrorString(error));                 \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
} while(0)

void cpuVerificationFunction(thrust::host_vector<int>& inputVector, thrust::host_vector<int>& predicateVector, thrust::host_vector<int>& outVector) {

	// Compute output length using the predicate vector
	int outLength = thrust::reduce(thrust::host, predicateVector.begin(), predicateVector.end());
	thrust::stable_partition(thrust::host, inputVector.begin(), inputVector.end(), predicateVector.begin(), ::cuda::std::identity{});

	// Copy subarray to output
	outVector.resize(outLength);
	thrust::copy(thrust::host, inputVector.begin(), inputVector.begin() + outLength, outVector.begin());
}

void launch() {

	// Setting input lengths
	int input_sizes[TEST_CASES] = { 5,10,20,38,53,64,73,85,91,100 };

	// Initiating random number generators
	std::minstd_rand randObj(RAND_SEED);
	std::uniform_int_distribution<int> inpDistObj(UNIFORM_DIST_LOWER_BOUND, UNIFORM_DIST_UPPER_BOUND);
	std::uniform_int_distribution<int> predicateDistObj(-50, 50);

	// Lambda functions to generate random numbers with different distribution parameters for input and predicate vectors
	auto getRandNum = [&](int x) {
		randObj.discard(x);
		return (int)inpDistObj(randObj);
		};

	auto getRandPred = [&](int x) {
		randObj.discard(x);
		return (int)(predicateDistObj(randObj) > 0);
		};

	for (int i = 0; i < TEST_CASES; i++) {

		int inp_length = input_sizes[i];

		// Host vectors
		thrust::host_vector<int> inputVector(inp_length);
		thrust::host_vector<int> predicateVector(inp_length);

		// Generate random input sequence and fill input and predicate vectors with random numbers
		thrust::tabulate(inputVector.begin(), inputVector.end(), getRandNum);
		thrust::tabulate(predicateVector.begin(), predicateVector.end(), getRandPred);

		// Device vectors
		thrust::device_vector<int> inputVector_d = inputVector;
		thrust::device_vector<int> predicateVector_d = predicateVector;

		// Calling implementation on GPU device
		thrust::device_vector<int> outArray_d;
		predicateBasedFilter(inputVector_d, predicateVector_d, outArray_d);

		// Copy output vector
		thrust::host_vector<int> outArray_device = outArray_d;
			
		// Calling implementation on CPU for verification
		thrust::host_vector<int> outArray_host;
		cpuVerificationFunction(inputVector, predicateVector, outArray_host);

		assert(outArray_device.size() == outArray_host.size());
		for (int i = 0; i < outArray_device.size(); i++)
			assert(outArray_device[i] == outArray_host[i]);
	}
}

void benchmark() {
    const int N = 10000000;

    std::minstd_rand randObj(RAND_SEED);
    std::uniform_int_distribution<int> inpDistObj(UNIFORM_DIST_LOWER_BOUND, UNIFORM_DIST_UPPER_BOUND);
    std::uniform_int_distribution<int> predicateDistObj(-50, 50);

    thrust::host_vector<int> h_input(N);
    thrust::host_vector<int> h_predicate(N);

    for (int i = 0; i < N; i++) {
        h_input[i] = inpDistObj(randObj);
    }
    for (int i = 0; i < N; i++) {
        h_predicate[i] = (predicateDistObj(randObj) > 0) ? 1 : 0;
    }

    thrust::device_vector<int> d_input_orig = h_input;
    thrust::device_vector<int> d_predicate = h_predicate;

    const int warmup = 3;
    const int timed = 100;

    for (int i = 0; i < warmup; i++) {
        thrust::device_vector<int> d_input = d_input_orig;
        thrust::device_vector<int> d_out;
        predicateBasedFilter(d_input, d_predicate, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    nvtxRangePushA("bench_region");
    for (int i = 0; i < timed; i++) {
        thrust::device_vector<int> d_input = d_input_orig;
        thrust::device_vector<int> d_out;
        predicateBasedFilter(d_input, d_predicate, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    nvtxRangePop();
}

int main(int argc, char** argv) {
	if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
		benchmark();
	} else {
		launch();
	}
}