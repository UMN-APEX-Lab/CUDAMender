#include <cuda_runtime.h>
#include "binary_search.h"

#ifndef SHARED_MEM_SIZE
#define SHARED_MEM_SIZE 256
#endif

__global__ void k_binarySearch(int *arr_d, int *queries_d, int *results_d, int arrSize, int querySize) {
    __shared__ int s_vals[SHARED_MEM_SIZE];
    __shared__ int s_idx[SHARED_MEM_SIZE];

    const int tid = threadIdx.x;
    const int blockThreads = blockDim.x;
    const int globalThread = blockIdx.x * blockThreads + tid;
    const int totalThreads = gridDim.x * blockThreads;

    if (arrSize <= SHARED_MEM_SIZE) {
        for (int i = tid; i < arrSize; i += blockThreads) {
            s_vals[i] = arr_d[i];
        }
        __syncthreads();

        for (int q = globalThread; q < querySize; q += totalThreads) {
            int target = queries_d[q];
            int lo = 0;
            int hi = arrSize - 1;
            int found = -1;

            while (lo <= hi) {
                int mid = lo + ((hi - lo) >> 1);
                int v = s_vals[mid];
                if (v == target) {
                    found = mid;
                    break;
                } else if (v < target) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }

            results_d[q] = found;
        }
        return;
    }

    const int splitterCount = SHARED_MEM_SIZE;
    for (int i = tid; i < splitterCount; i += blockThreads) {
        int idx = ((long long)(i + 1) * arrSize) / (splitterCount + 1);
        if (idx >= arrSize) idx = arrSize - 1;
        s_idx[i] = idx;
        s_vals[i] = arr_d[idx];
    }
    __syncthreads();

    for (int q = globalThread; q < querySize; q += totalThreads) {
        int target = queries_d[q];

        int slo = 0;
        int shi = splitterCount - 1;
        int pos = splitterCount;

        while (slo <= shi) {
            int smid = slo + ((shi - slo) >> 1);
            int v = s_vals[smid];
            if (v >= target) {
                pos = smid;
                shi = smid - 1;
            } else {
                slo = smid + 1;
            }
        }

        int lo, hi;
        if (pos == 0) {
            lo = 0;
            hi = s_idx[0];
        } else if (pos == splitterCount) {
            lo = s_idx[splitterCount - 1] + 1;
            hi = arrSize - 1;
        } else {
            lo = s_idx[pos - 1] + 1;
            hi = s_idx[pos];
        }

        if (pos < splitterCount && s_vals[pos] == target) {
            results_d[q] = s_idx[pos];
            continue;
        }

        int found = -1;
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            int v = arr_d[mid];
            if (v == target) {
                found = mid;
                break;
            } else if (v < target) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }

        results_d[q] = found;
    }
}