/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


// Atomic-max function
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data,
    int padded_length) {

    // Point-wise multiplication and scaling for the 
    // FFT'd input and impulse response. 

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < padded_length) {
        // Read the two factors
        cufftComplex v1 = raw_data[i];
        cufftComplex v2 = impulse_v[i];

        cufftComplex v3;
        // Get the real component of the product
        v3.x = v1.x * v2.x - v1.y * v2.y,
        // Get the imaginary component of the product
        v3.y = v1.x * v2.y + v1.y * v2.x;
        // Scale
        v3.x /= padded_length;
        v3.y /= padded_length;
        // Write the product back
        out_data[i] = v3;

        // Move to next set of blocks to be processed
        i += gridDim.x * blockDim.x;
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* Find the maximum real value of the out_data array */

    extern __shared__ float smem[]; // Store warp maxes here.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float val = 0;

    while (i < padded_length) {
        /* 
         * Reduction Info:
         * First we reduce within a warp using warp shuffle. This is fast
         * because it uses registers and doesn't require reading/writing to
         * memory.
         *
         * Then we write these values into shared memory using the sequential
         * addressing parallel reduction given in Mark Harris's
         * "Optimizing Parallel Reduction in CUDA" presentation.
         */

        /*
         * Read from out_data, 
         * find the max out of the warp,
         * then store the warp value in smem.
         */

        val = fabsf(out_data[i].x);
        int warpIdx = threadIdx.x >> 5;

        // Butterfly warp shuffle pattern to get max of warp items
        for (int j = 16; j >= 1; j /= 2) {
            float otherval = __shfl_xor(val, j, 32);
            val = (val > otherval) ? val : otherval;
        }

        smem[warpIdx] = val;

        __syncthreads();
        
        // Reduce values in smem using sequential addressing
        // to avoid bank conflicts
        for (unsigned int s = blockDim.x/64; s > 0; s >>= 1) {
            if (tid < s) {
                smem[tid] = fmaxf(smem[tid], smem[tid + s]);
            }
            __syncthreads();
        }

        // Fold max of block values into max_abs_val atomically
        if (tid == 0)
            atomicMax(max_abs_val, smem[0]);
        // Move to next set of blocks to be processed
        i += gridDim.x * blockDim.x;
   } 

}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    // Divide all data by the value pointed to by max_abs_val.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < padded_length) {
        // Read the value to be divided and the divisor
        float max_val = *max_abs_val;
        cufftComplex val = out_data[i];
        
        // Divide real and imaginary components
        val.x /= max_val;
        val.y /= max_val;

        // Write back
        out_data[i] = val;

        // Move to next set of blocks to be processed
        i += gridDim.x * blockDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(
        raw_data,
        impulse_v,
        out_data,
        padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    cudaMaximumKernel<<<blocks, threadsPerBlock, 
                        (threadsPerBlock * sizeof(float)) / 32>>> (
        out_data, max_abs_val, padded_length);
}

void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    cudaDivideKernel<<<blocks, threadsPerBlock>>> (
        out_data,
        max_abs_val,
        padded_length);
}
