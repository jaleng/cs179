/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve_cuda.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
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


    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < padded_length) {
        cufftComplex v1 = raw_data[i];
        cufftComplex v2 = impulse_v[i];
        cufftComplex v3;
        v3.x = v1.x * v2.x - v1.y * v2.y,
        v3.y = v1.x * v2.y + v1.y * v2.x;
        v3.x /= padded_length;
        v3.y /= padded_length;

        out_data[i] = v3;

        i += gridDim.x * blockDim.x;
    }
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding and subsequent
    normalization (dividing by maximum).

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */
    extern __shared__ float smem[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < padded_length) {

        float val = out_data[i].x;

        // Butterfly warp shuffle pattern to get max of warp items
        for (int j = 16; j >= 1; j /= 2) {
            float otherval = __shfl_xor(val, j, 32);
            val = (val > otherval) ? val : otherval;
        }

        int warpIdx = i >> 5;
        smem[warpIdx] = val;
        i += gridDim.x * blockDim.x;
    }
    
    __syncthreads();

    i = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int s_idx = blockDim.x/2; s_idx > 0; s_idx >>= 1) {
        if (i < s_idx) {
            float v1 = smem[i];
            float v2 = smem[i + s_idx];
            smem[i] = (v1 > v2) ? v1 : v2;
        }
        __syncthreads();
    }

    if (i == 0)
        atomicMax(max_abs_val, smem[0]);
}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < padded_length) {
        float max_val = *max_abs_val;
        cufftComplex val = out_data[i];
        val.x /= max_val;
        val.y /= max_val;
        out_data[i] = val;

        i += gridDim.x * blockDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* TODO: Call the element-wise product and scaling kernel. */
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
        

    /* TODO 2: Call the max-finding kernel. */
    cudaMaximumKernel<<<blocks, threadsPerBlock, (padded_length * sizeof(float)) / 32>>> (
        out_data, max_abs_val, padded_length);

}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */
    cudaDivideKernel<<<blocks, threadsPerBlock>>> (
        out_data,
        max_abs_val,
        padded_length);
}
