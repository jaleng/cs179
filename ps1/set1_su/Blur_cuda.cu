/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>

#include "Blur_cuda.cuh"


__global__
void
cudaBlurKernel(const float *raw_data, const float *blur_v, float *out_data,
    int N, int blur_v_size) {

    /* TODO: Fill in the implementation for the GPU-accelerated convolution. 

    It may be helpful to use the information in the lecture slides, 
    as well as the CPU implementation, as a reference. */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < N) {
        if (i < blur_v_size) {
            for (int j = 0; j <= i; ++j) {
                out_data[i] = raw_data[i - j] * blur_v[j];
            }
        } 
        else {
            for (int j = 0; j < blur_v_size; ++j)
            {
                out_data[i] = raw_data[i - j] * blur_v[j];
            }
        }

        i += gridDim.x * blockDim.x;
    }
}


void cudaCallBlurKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const float *raw_data,
        const float *blur_v,
        float *out_data,
        const unsigned int N,
        const unsigned int blur_v_size) {
        
    /* TODO: Call the kernel above this function. */
    cudaBlurKernel<<<blocks, threadsPerBlock>>>(raw_data, blur_v, out_data, N, blur_v_size);
}
