#ifndef K_SORT_SEGMENTS_H
#define K_SORT_SEGMENTS_H

__global__ void k_sortSegments(float *array_d, float *arrayOut_d, int segmentSize, int arraySize);

#endif // K_SORT_SEGMENTS_H