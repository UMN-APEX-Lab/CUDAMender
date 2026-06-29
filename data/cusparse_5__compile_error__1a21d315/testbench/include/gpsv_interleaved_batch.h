#ifndef GPSV_INTERLEAVED_BATCH_H
#define GPSV_INTERLEAVED_BATCH_H

#include <vector>

void solveGpsvInterleavedBatch(
    int n, int batchSize, const std::vector<float> &h_S,
    const std::vector<float> &h_L, const std::vector<float> &h_M,
    const std::vector<float> &h_U, const std::vector<float> &h_W,
    const std::vector<float> &h_B, std::vector<float> &h_X);

#endif // GPSV_INTERLEAVED_BATCH_H