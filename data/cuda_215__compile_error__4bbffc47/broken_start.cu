#include <cuda_runtime.h>

__global__ void k_computeLBP(const unsigned char* input_d, unsigned char* output_d, int width, int height) {
    // Fixed width is 1024 as per description.
    // We use a tile size for shared memory. Let's assume blockDim is (32, 32).
    // To handle halo, we need (blockDim.x + 2) * (blockDim.y + 2).
    
    extern __shared__ unsigned char s_img[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;

    // Calculate shared memory indices
    // s_img is indexed as [ty + 1][tx + 1] for the interior
    int s_width = blockDim.x + 2;
    
    // Grid-stride loop for height/width if necessary, 
    // but usually, LBP is mapped 1:1 to pixels.
    // The prompt mentions grid-stride loop pattern, but usually, 
    // image kernels are mapped by blocks.
    
    int x = bx + tx;
    int y = by + ty;

    // Load data into shared memory including halo
    // Each thread loads its primary pixel and helps load the halo
    if (x < width && y < height) {
        s_img[(ty + 1) * s_width + (tx + 1)] = input_d[y * width + x];
    } else {
        s_img[(ty + 1) * s_width + (tx + 1)] = 0;
    }

    // Load Top Halo
    if (ty == 0) {
        int load_y = y - 1;
        for (int i = tx; i < s_width; i += blockDim.x) {
            int load_x = bx + i - 1;
            if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height)
                s_img[0 * s_width + i] = input_d[load_y * width + load_x];
            else
                s_img[0 * s_width + i] = 0;
        }
    }
    // Load Bottom Halo
    if (ty == blockDim.y - 1) {
        int load_y = y + 1;
        for (int i = tx; i < s_width; i += blockDim.x) {
            int load_x = bx + i - 1;
            if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height)
                s_img[(blockDim.y + 1) * s_width + i] = input_d[load_y * width + load_x];
            else
                s_img[(blockDim.y + 1) * s_width + i] = 0;
        }
    }
    // Load Left Halo
    if (tx == 0) {
        for (int j = 0; j < blockDim.y + 2; ++j) {
            int load_y = by + j - 1;
            int load_x = bx - 1;
            if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height)
                s_img[j * s_width + 0] = input_d[load_y * width + load_x];
            else
                s_img[j * s_width + 0] = 0;
        }
    }
    // Load Right Halo
    if (tx == blockDim.x - 1) {
        for (int j = 0; j < blockDim.y + 2; ++j) {
            int load_y = by + j - 1;
            int load_x = bx + blockDim.x;
            if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height)
                s_img[j * s_width + (s_width - 1)] = input_d[load_y * width + load_x];
            else
                s_img[j * s_width + (s_width - 1)] = 0;
        }
    }

    __syncthreads();

    if (x >= 0 && x < width && y >= 0 && y < height) {
        if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
            output_d[y * width + x] = 0;
        } else {
            unsigned char center = s_img[(ty + 1) * s_width + (tx + 1)];
            unsigned char code = 0;

            // Clockwise starting from top-left
            // 0: TL, 1: T, 2: TR, 3: R, 4: BR, 5: B, 6: BL, 7: L
            code |= (s_img[ty * s_width + tx] >= center) << 0;
            code |= (s_img[ty * s_width + (tx + 1)] >= center) << 1;
            code |= (s_img[ty * s_width + (tx + 2)] >= center) << 2;
            code |= (s_img[(ty + 1) * s_width + (tx + 2)] >= center) << 3;
            code |= (s_img[(ty + 2) * s_width + (tx + 2)] >= center) << 4;
            code |= (s_img[(ty + 2) * s_width + (tx + 1)] >= center) << 5;
            code |= (s_img[(ty + 2) * s_width + tx] >= center) << 6;
            code |= (s_img[(ty + 1) * s_width + tx] >= center) << 7;

            output_d[y * width + x] = code;
        }
    }
}