#include "lbp_kernel.h"
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

struct TestCase {
    int widthOfImage;      
    int heightOfImage;
    std::vector<unsigned char> inputImage;         
    std::vector<unsigned char> expectedOutputImage;
    TestCase(int width, int height, const std::vector<unsigned char>& input, 
             const std::vector<unsigned char>& expected) : widthOfImage(width), heightOfImage(height), inputImage(input), expectedOutputImage(expected) {}
};

void launch(){
    // Create a CUDA stream for asynchronous operations.
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::vector<TestCase> testCases = {
        // 5x3 non-square image.
        TestCase(
            5, 3,
            {10, 10, 10, 10, 10,
             20, 20, 20, 20, 20,
             30, 30, 30, 30, 30},
            {131, 143, 143, 143, 14,
             128, 128, 128, 128, 0,
             128, 128, 128, 128, 0}
        ),
        
        // 3x2 non-square image.
        TestCase(
            3, 2,
            {50, 60, 70,
             80, 90, 100},
            {131, 135, 6, 
             128, 128, 0}
        ),

        // 2x5 non-square image.
        TestCase(
            2, 5,
            {10, 20,
             30, 40,
             50, 60,
             70, 80,
             90, 100},
            {131, 6,
             131, 6,
             128, 0, 
             131, 6, 
             128, 0}
        ),

        // 5x5 image.
        TestCase(
            5, 5,
            {10, 10, 10, 10, 10,
             10, 20, 20, 20, 10,
             10, 20, 30, 20, 10,
             10, 20, 20, 20, 10,
             10, 10, 10, 10, 10},
            {131, 143, 143, 143, 14, 
             195, 131, 143, 14, 62, 
             192, 224, 0, 56, 56, 
             131, 128, 136, 8, 14, 
             192, 240, 240, 240, 48}
        ),

        // 3x3 image.
        TestCase(
            3, 3,
            {5, 5, 5,
             5, 10, 5,
             5, 5, 5},
            {131, 143, 14,
             192, 0, 56, 
             128, 128, 0}
        ),

        // 4x4 image.
        TestCase(
            4, 4,
            {1, 2, 3, 4,
             5, 6, 7, 8,
             9, 10, 11, 12,
             13, 14, 15, 16},
            {135, 135, 135, 6,
             128, 128, 128, 0,
             131, 135, 135, 6,
             128, 128, 128, 0}
        ),
        
        // 6x6 constant image.
        TestCase(
            6, 6,
            {50, 50, 50, 50, 50, 50,
             50, 50, 50, 50, 50, 50,
             50, 50, 50, 50, 50, 50,
             50, 50, 50, 50, 50, 50,
             50, 50, 50, 50, 50, 50,
             50, 50, 50, 50, 50, 50},
            {131, 143, 143, 143, 143, 14,
             195, 255, 255, 255, 255, 62,
             192, 240, 240, 240, 240, 48, 
             131, 143, 143, 143, 143, 14, 
             195, 255, 255, 255, 255, 62,
             192, 240, 240, 240, 240, 48}
        ),
        
        // 7x7 image.
        TestCase(
            7, 7,
            {10, 10, 10, 10, 10, 10, 10,
             20, 20, 20, 20, 20, 20, 20,
             30, 30, 30, 30, 30, 30, 30,
             40, 40, 40, 40, 40, 40, 40,
             50, 50, 50, 50, 50, 50, 50,
             60, 60, 60, 60, 60, 60, 60,
             70, 70, 70, 70, 70, 70, 70},
            {131, 143, 143, 143, 143, 143, 14,
             131, 143, 143, 143, 143, 143, 14, 
             128, 128, 128, 128, 128, 128, 0,
             131, 143, 143, 143, 143, 143, 14,
             128, 128, 128, 128, 128, 128, 0,
             131, 143, 143, 143, 143, 143, 14,
             128, 128, 128, 128, 128, 128, 0}
        )
    };

    int totalTestCases = testCases.size();

    // Allocate device memory once based on maximum image size.
    int maxWidth = 0, maxHeight = 0;
    for (int i = 0; i < totalTestCases; i++) {
         maxWidth = std::max(maxWidth, testCases[i].widthOfImage);
         maxHeight = std::max(maxHeight, testCases[i].heightOfImage);
    }
    int maxNumPixels = maxWidth * maxHeight;
    size_t maxImgSize = maxNumPixels * sizeof(unsigned char);
    unsigned char *input_d, *output_d;
    CUDA_CHECK(cudaMallocAsync(&input_d, maxImgSize, stream));
    CUDA_CHECK(cudaMallocAsync(&output_d, maxImgSize, stream));

    // Launch Kernel
    auto launchKernelWithConfig = [&stream](
        const unsigned char* segmentInput_d, unsigned char* segmentOutput_d,
        int segmentSize, int width, int segHeight,
        int numNeighbors, float radius,
        int threadsPerBlock, int blocksPerGrid) {
        
        // Convert to thread blocks
        int blockDimX = min(MAX_BLOCKS_PER_SEGMENT, threadsPerBlock);
        int blockDimY = max(1, threadsPerBlock / blockDimX);

        // Shared memory size calculation from dynamicSMemSizeFunc
        size_t sharedMemSize = dynamicSMemSizeFunc(threadsPerBlock);
        
        // Kernel Launch
        void* args[] = { 
            (void*)&segmentInput_d, 
            (void*)&segmentOutput_d, 
            (void*)&width, 
            (void*)&segHeight, 
            (void*)&numNeighbors, 
            (void*)&radius 
        };
        CUDA_CHECK(cudaLaunchKernel((void*)k_localBinaryKernel,
                                    dim3(blocksPerGrid, 1, 1),
                                    dim3(blockDimX, blockDimY, 1),
                                    args, sharedMemSize, stream));
    };
    
    // Run test with multiple kernel launches for segmentation.
    auto runTestWithMultipleLaunches = [&](
        const unsigned char* input, const unsigned char* expected,

        // Total number of pixels, image width, image height , neighbouring sample and radius.
        int numPixels, int width, int height,
        int numNeighbors, float radius) {

        // Allocate device memory
        CUDA_CHECK(cudaMemcpyAsync(input_d, input, numPixels * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(output_d, 0, numPixels * sizeof(unsigned char), stream));
        
        // Divide the image into segments by rows.
        int numRows = height;

        // Choose number of segments based on row count.
        int numSegments = (numRows <= 2) ? 1 : ((numRows <= 6) ? 2 : 3);
        int baseRows = numRows / numSegments;
        int remainderRows = numRows % numSegments;
        
        // Starting row index for the current segment.
        int startRow = 0;
        for (int i = 0; i < numSegments; i++) {

            // Number of rows in this segment.
            int segmentRows = baseRows + (i < remainderRows ? 1 : 0);
            // Total pixels in this segment.
            int segmentSize = segmentRows * width;
            
            const unsigned char* segmentInput_d = input_d + startRow * width;
            unsigned char* segmentOutput_d = output_d + startRow * width;
            
            // Launch configuration for segment.
            // Use occupancy API to determine optimal block size for segment.   
            int optBlockSize, blocksPerGrid;
            float theoreticalOccupancy;
            CUDA_CHECK(getOptimalLaunchParams(segmentSize, optBlockSize, blocksPerGrid, theoreticalOccupancy));

            // Set threads per block to optimal value, but not more than segment size.
            int threadsPerBlock = (segmentSize < optBlockSize) ? segmentSize : optBlockSize;

            // Ensure minimum thread count for very small images
            threadsPerBlock = max(threadsPerBlock, MAX_BLOCKS_PER_SEGMENT);
            
            // Launch kernel on the current segment
            launchKernelWithConfig(segmentInput_d, segmentOutput_d, segmentSize, width, segmentRows, numNeighbors, radius, threadsPerBlock, blocksPerGrid);

            // Update starting row for next segment.
            startRow += segmentRows;
        }
        
        // Allocate host output buffer and copy the device output to host.
        std::vector<unsigned char> output_h(numPixels);
        CUDA_CHECK(cudaMemcpyAsync(output_h.data(), output_d, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Verify results
        for (int i = 0; i < numPixels; i++) {
            assert(output_h[i] == expected[i]);
        }
    };
    auto runTest = [&] (const TestCase &tc) {
        int numPixels = tc.widthOfImage * tc.heightOfImage;
        runTestWithMultipleLaunches(tc.inputImage.data(), tc.expectedOutputImage.data(), numPixels, tc.widthOfImage,                   tc.heightOfImage, 8, 1.0f);
    };
   
    for (int i = 0; i < totalTestCases; i++) {
        runTest(testCases[i]);
    }
        
    // Cleanup
    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream)); 
}

void benchmark() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const int width = 2048;
    const int height = 2048;
    const int numPixels = width * height;
    const int numNeighbors = 8;
    const float radius = 1.0f;

    std::vector<unsigned char> input_h(numPixels);
    for (int i = 0; i < numPixels; i++) {
        input_h[i] = (unsigned char)(i % 256);
    }

    unsigned char *input_d, *output_d;
    CUDA_CHECK(cudaMallocAsync(&input_d, numPixels * sizeof(unsigned char), stream));
    CUDA_CHECK(cudaMallocAsync(&output_d, numPixels * sizeof(unsigned char), stream));
    CUDA_CHECK(cudaMemcpyAsync(input_d, input_h.data(), numPixels * sizeof(unsigned char), cudaMemcpyHostToDevice, stream));

    auto runOneIteration = [&]() {
        CUDA_CHECK(cudaMemsetAsync(output_d, 0, numPixels * sizeof(unsigned char), stream));
        int numRows = height;
        int numSegments = 3;
        int baseRows = numRows / numSegments;
        int remainderRows = numRows % numSegments;
        int startRow = 0;
        for (int s = 0; s < numSegments; s++) {
            int segmentRows = baseRows + (s < remainderRows ? 1 : 0);
            int segmentSize = segmentRows * width;
            const unsigned char* segmentInput_d = input_d + startRow * width;
            unsigned char* segmentOutput_d = output_d + startRow * width;
            int optBlockSize, blocksPerGrid;
            float theoreticalOccupancy;
            CUDA_CHECK(getOptimalLaunchParams(segmentSize, optBlockSize, blocksPerGrid, theoreticalOccupancy));
            int threadsPerBlock = (segmentSize < optBlockSize) ? segmentSize : optBlockSize;
            threadsPerBlock = max(threadsPerBlock, MAX_BLOCKS_PER_SEGMENT);
            int blockDimX = min(MAX_BLOCKS_PER_SEGMENT, threadsPerBlock);
            int blockDimY = max(1, threadsPerBlock / blockDimX);
            size_t sharedMemSize = dynamicSMemSizeFunc(threadsPerBlock);
            void* args[] = {
                (void*)&segmentInput_d,
                (void*)&segmentOutput_d,
                (void*)&width,
                (void*)&segmentRows,
                (void*)&numNeighbors,
                (void*)&radius
            };
            CUDA_CHECK(cudaLaunchKernel((void*)k_localBinaryKernel,
                                        dim3(blocksPerGrid, 1, 1),
                                        dim3(blockDimX, blockDimY, 1),
                                        args, sharedMemSize, stream));
            startRow += segmentRows;
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

    // Warmup
    for (int i = 0; i < 3; i++) {
        runOneIteration();
    }

    // Timed region
    nvtxRangePushA("bench_region");
    for (int i = 0; i < 100; i++) {
        runOneIteration();
    }
    nvtxRangePop();

    CUDA_CHECK(cudaFreeAsync(input_d, stream));
    CUDA_CHECK(cudaFreeAsync(output_d, stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--perf") == 0) {
        benchmark();
    } else {
        launch();
    }
}