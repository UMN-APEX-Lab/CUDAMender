#include "kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void k_calculateElement(uint32_t booleanElementsPerInteger, uint32_t numIntegers, uint32_t* integersIn_d, uint32_t* integersOut_d) {
    constexpr uint32_t INTERIOR_30_ELEMENTS_MASK = 0b01111111111111111111111111111110;
    constexpr uint32_t BORDER_LEFT_ELEMENT_MASK = 0b00000000000000000000000000000001;
    constexpr uint32_t BORDER_RIGHT_ELEMENT_MASK = 0b10000000000000000000000000000000;
    uint32_t globalThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t totalThreads = gridDim.x * blockDim.x;
    uint32_t numGridStrideSteps = (numIntegers + (totalThreads * FOUR_INTEGERS_PER_THREAD) - 1) / (totalThreads * FOUR_INTEGERS_PER_THREAD);
    // Grid-stride loop to load four integers (representing 128 booleans), compute in chunks of 128 booleans (using bitwise parallelism for each integer), and store four integers per thread at once.
    for (int step = 0; step < numGridStrideSteps; step++) {
        int index = step * (totalThreads * FOUR_INTEGERS_PER_THREAD) + globalThreadIndex * FOUR_INTEGERS_PER_THREAD;
        uint4 leftIntegers = ((index - (int)FOUR_INTEGERS_PER_THREAD >= 0 && index - FOUR_INTEGERS_PER_THREAD < numIntegers) ? *reinterpret_cast<uint4*>(&integersIn_d[index - FOUR_INTEGERS_PER_THREAD]) : make_uint4(0, 0, 0, 0));
        uint4 centerIntegers = ((index < numIntegers) ? *reinterpret_cast<uint4*>(&integersIn_d[index]) : make_uint4(0, 0, 0, 0));
        uint4 rightIntegers = ((index + FOUR_INTEGERS_PER_THREAD < numIntegers) ? *reinterpret_cast<uint4*>(&integersIn_d[index + FOUR_INTEGERS_PER_THREAD]) : make_uint4(0, 0, 0, 0));
        uint4 results = make_uint4(0, 0, 0, 0);
        for (int i = 0; i < FOUR_INTEGERS_PER_THREAD; i++) {
            uint32_t leftInteger;
            uint32_t centerInteger;
            uint32_t rightInteger;
            switch(i) {
                case 0: { 
                    leftInteger = leftIntegers.w; 
                    centerInteger = centerIntegers.x; 
                    rightInteger = centerIntegers.y; 
                    break;
                }
                case 1: { 
                    leftInteger = centerIntegers.x; 
                    centerInteger = centerIntegers.y; 
                    rightInteger = centerIntegers.z; 
                    break;
                }
                case 2: { 
                    leftInteger = centerIntegers.y; 
                    centerInteger = centerIntegers.z; 
                    rightInteger = centerIntegers.w; 
                    break;
                }
                case 3: { 
                    leftInteger = centerIntegers.z; 
                    centerInteger = centerIntegers.w; 
                    rightInteger = rightIntegers.x; 
                    break;
                }
                default: { 
                    break;
                }
            }
            
            uint32_t interiorLeftNeighbors = centerInteger << 1;
            uint32_t interiorRightNeighbors = centerInteger >> 1;
            uint32_t interiorResult = interiorLeftNeighbors ^ centerInteger ^ interiorRightNeighbors;
            interiorResult = interiorResult & INTERIOR_30_ELEMENTS_MASK;
            
            uint32_t leftResult = (leftInteger >> (booleanElementsPerInteger - 1)) ^ centerInteger ^ interiorRightNeighbors;
            leftResult = leftResult & BORDER_LEFT_ELEMENT_MASK;
            
            uint32_t rightResult = interiorLeftNeighbors ^ centerInteger ^ ((rightInteger & 1) << (booleanElementsPerInteger - 1));
            rightResult = rightResult & BORDER_RIGHT_ELEMENT_MASK;
            
            uint32_t result = leftResult | interiorResult | rightResult;
            
            if (index + i < numIntegers) {
                switch(i) {
                    case 0: { 
                        results.x = result;
                        break;
                    }
                    case 1: { 
                        results.y = result; 
                        break;
                    }
                    case 2: { 
                        results.z = result; 
                        break;
                    }
                    case 3: { 
                        results.w = result; 
                        break;
                    }
                    default: { 
                        break;
                    }
                }
            }
        }
        if (index < numIntegers) {
            *reinterpret_cast<uint4*>(&integersOut_d[index]) = results;
        }
    }
}