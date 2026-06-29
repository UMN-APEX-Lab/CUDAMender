#include <cufft.h>
#include <cufftXt.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <complex>
#include "fft_mgpu.h"
#include "fft_helpers.h"

void fft_3d_mgpu_r2c_c2r_example(std::array<int, 3> &dims,
                                 std::vector<int> &gpus,
                                 std::vector<std::complex<float>> &h_data_in,
                                 std::vector<std::complex<float>> &h_data_out) {
    int nx = dims[0];
    int ny = dims[1];
    int nz = dims[2];
    
    int nGPUs = gpus.size();
    
    // Create cuFFT plans for multi-GPU execution
    cufftHandle plan_r2c, plan_c2r;
    CUFFT_CHECK(cufftCreate(&plan_r2c));
    CUFFT_CHECK(cufftCreate(&plan_c2r));
    
    // Set the GPUs for multi-GPU execution
    CUFFT_CHECK(cufftXtSetGPUs(plan_r2c, nGPUs, gpus.data()));
    CUFFT_CHECK(cufftXtSetGPUs(plan_c2r, nGPUs, gpus.data()));
    
    // Create the 3D FFT plans
    // For R2C: nx * ny * nz real -> nx * ny * (nz/2+1) complex
    size_t workSize_r2c, workSize_c2r;
    CUFFT_CHECK(cufftMakePlan3d(plan_r2c, nx, ny, nz, CUFFT_R2C, &workSize_r2c));
    CUFFT_CHECK(cufftMakePlan3d(plan_c2r, nx, ny, nz, CUFFT_C2R, &workSize_c2r));
    
    // Allocate device memory descriptor using cufftXtMalloc
    cudaLibXtDesc *desc;
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &desc, CUFFT_XT_FORMAT_INPLACE));
    
    // Prepare host input data as float (real) for R2C transform
    // The input is nx*ny*nz real values
    size_t real_size = nx * ny * nz;
    std::vector<float> h_real_in(real_size);
    for (size_t i = 0; i < real_size; ++i) {
        h_real_in[i] = h_data_in[i].real();
    }
    
    // Copy input data from host to device using cufftXtMemcpy
    // Set up host descriptor for input
    cudaLibXtDesc *h_desc_in;
    CUFFT_CHECK(cufftXtMalloc(plan_r2c, &h_desc_in, CUFFT_XT_FORMAT_INPLACE));
    
    // Copy real data to the host descriptor
    for (int i = 0; i < nGPUs; ++i) {
        CUDA_CHECK(cudaSetDevice(gpus[i]));
        size_t offset = i * (real_size / nGPUs + (i < (real_size % nGPUs) ? 1 : 0));
        size_t count = real_size / nGPUs + (i < (real_size % nGPUs) ? 1 : 0);
        if (count > 0 && offset < real_size) {
            size_t actual_count = std::min(count, real_size - offset);
            CUDA_CHECK(cudaMemcpy(h_desc_in->descriptor->data[i], 
                                 h_real_in.data() + offset, 
                                 actual_count * sizeof(float), 
                                 cudaMemcpyHostToDevice));
        }
    }
    
    // Copy from host descriptor to device descriptor
    CUFFT_CHECK(cufftXtMemcpy(plan_r2c, desc, h_desc_in, CUFFT_COPY_HOST_TO_DEVICE));
    
    // Free the temporary host descriptor
    CUFFT_CHECK(cufftXtFree(h_desc_in));
    
    // Execute R2C FFT plan
    CUFFT_CHECK(cufftXtExecDescriptor(plan_r2c, desc, desc, CUFFT_FORWARD));
    
    // Scale the complex results by factor of 2.0
    scaleComplex(desc, 2.0f, nx, ny, nz / 2 + 1, nGPUs, gpus);
    
    // Execute C2R FFT plan
    CUFFT_CHECK(cufftXtExecDescriptor(plan_c2r, desc, desc, CUFFT_INVERSE));
    
    // Copy results back to host
    // Set up host descriptor for output
    cudaLibXtDesc *h_desc_out;
    CUFFT_CHECK(cufftXtMalloc(plan_c2r, &h_desc_out, CUFFT_XT_FORMAT_INPLACE));
    
    // Copy from device descriptor to host descriptor
    CUFFT_CHECK(cufftXtMemcpy(plan_c2r, h_desc_out, desc, CUFFT_COPY_DEVICE_TO_HOST));
    
    // Copy real data from host descriptor to output vector
    std::vector<float> h_real_out(real_size);
    for (int i = 0; i < nGPUs; ++i) {
        CUDA_CHECK(cudaSetDevice(gpus[i]));
        size_t offset = i * (real_size / nGPUs + (i < (real_size % nGPUs) ? 1 : 0));
        size_t count = real_size / nGPUs + (i < (real_size % nGPUs) ? 1 : 0);
        if (count > 0 && offset < real_size) {
            size_t actual_count = std::min(count, real_size - offset);
            CUDA_CHECK(cudaMemcpy(h_real_out.data() + offset,
                                 h_desc_out->descriptor->data[i],
                                 actual_count * sizeof(float),
                                 cudaMemcpyDeviceToHost));
        }
    }
    
    // Convert float results to complex<float> output
    for (size_t i = 0; i < real_size; ++i) {
        h_data_out[i] = std::complex<float>(h_real_out[i], 0.0f);
    }
    
    // Clean up resources
    CUFFT_CHECK(cufftXtFree(h_desc_out));
    CUFFT_CHECK(cufftXtFree(desc));
    CUFFT_CHECK(cufftDestroy(plan_r2c));
    CUFFT_CHECK(cufftDestroy(plan_c2r));
}