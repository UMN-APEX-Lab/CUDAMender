#ifndef LOG2_KERNEL_H
#define LOG2_KERNEL_H

// Number of elements in the lookup table for log2(mantissa).
constexpr int LOOKUP_TABLE_ELEMENTS = 256;

__global__ void k_calculateLog2(int numElements, float * data_d, float * lookupTable_d, int lookupTableDuplication);

#endif // LOG2_KERNEL_H