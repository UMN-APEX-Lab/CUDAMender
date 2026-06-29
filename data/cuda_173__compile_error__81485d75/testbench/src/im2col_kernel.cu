#include "im2col_kernel.h"

__global__ void k_im2ColTransform(__nv_bfloat16 *input,
                                  __nv_bfloat16 *weights,
                                  __nv_bfloat16 *im2ColPad,
                                  __nv_bfloat16 *weightPad,
                                  int channels,
                                  int inputHeight,
                                  int inputWidth,
                                  int kernelHeight,
                                  int kernelWidth,
                                  int pad,
                                  int stride,
                                  int outHeight,
                                  int outWidth,
                                  int numFilters,
                                  int padK) {
    // Derived dimensions.
    int weightK = channels * kernelHeight * kernelWidth;   // Unpadded reduction dimension.
    int im2colRows = weightK;                                // Each row corresponds to (c, kh, kw).
    int im2colCols = outHeight * outWidth;                   // Each column corresponds to an output spatial location.

    // Total iterations for weight processing and im2col processing.
    int totalWeight = numFilters * padK;         // Process padded weights.
    int totalIm2col = padK * im2colCols;           // Process padded im2col.
    int total = totalWeight + totalIm2col;         // Combined total.

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total)
        return;

    //Weight Processing -> Flatten & Pad
    if (tid < totalWeight) {
        int filter = tid / padK;      // Which filter.
        int j = tid % padK;           // Index in padded reduction dimension.
        if (j < weightK) {
            // Copy original weight element.
            weightPad[tid] = weights[filter * weightK + j];
        } else {
            // Pad with zero.
            weightPad[tid] = __nv_bfloat16(0.0f);
        }
    }
    //im2col transformation and Padding
    else {
        int index = tid - totalWeight;  // Index into im2col padded space.
        int row = index / im2colCols;     // Row in the padded im2col matrix.
        int col = index % im2colCols;     // Column index.
        if (row < im2colRows) {
            // Determine (channel, kh, kw) indices.
            int c = row / (kernelHeight * kernelWidth);
            int rem = row % (kernelHeight * kernelWidth);
            int kh = rem / kernelWidth;
            int kw = rem % kernelWidth;
            int outY = col / outWidth;
            int outX = col % outWidth;
            int inputY = outY * stride - pad + kh;
            int inputX = outX * stride - pad + kw;
            __nv_bfloat16 val = __nv_bfloat16(0.0f);
            if (inputY >= 0 && inputY < inputHeight && inputX >= 0 && inputX < inputWidth) {
                int inputIndex = c * (inputHeight * inputWidth) + inputY * inputWidth + inputX;
                val = input[inputIndex];
            }
            im2ColPad[index] = val;
        } else {
            // Pad rows beyond the original im2col rows.
            im2ColPad[index] = __nv_bfloat16(0.0f);
        }
    }
}