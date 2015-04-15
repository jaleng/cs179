#include <cassert>
#include <cuda_runtime.h>
#include "transpose_cuda.cuh"

/**
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
*/


/**
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304  matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
  // TODO: do not modify code, just comment on suboptimal accesses

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  for (; j < end_j; j++) {
    // write output is non-coalesced, accesses are n floats apart,
    // so each will hit a new cache line. (32 cache lines touched).
    output[j + n * i] = input[i + n * j];
  }
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
  // TODO: Modify transpose kernel to use shared memory. All global memory
  // reads and writes should be coalesced. Minimize the number of shared
  // memory bank conflicts (0 bank conflicts should be possible using
  // padding). Again, comment on all sub-optimal accesses.

  __shared__ float data[4160];

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;
  int dj = 4 * threadIdx.y;
  int di = threadIdx.x;

  for (; j < end_j; j++, dj++) {
    //output[j + n * i] = input[i + n * j];
    data[dj + (65 * di)] = input[i + n * j];
  }
  __syncthreads();

  j = 4 * threadIdx.y + 64 * blockIdx.y;
  dj = 4 * threadIdx.y;
  int j0 = 64 * blockIdx.y;
  int i0 = 64 * blockIdx.x;
  for (; j < end_j; j++, dj++) {
    //output[j + n * i] = input[i + n * j];
    output[j0 + n * i0 + di + n * dj] = data[di + (65 * dj)];
  }


}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
  // TODO: This should be based off of your shmemTransposeKernel.
  // Use any optimization tricks discussed so far to improve performance.
  // Consider ILP and loop unrolling.

  __shared__ float data[4160];

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  int dj = 4 * threadIdx.y;
  int di = threadIdx.x;

  int i_idx1 = i + n * j;
  int i_idx2 = i + n * (j + 1);
  int i_idx3 = i + n * (j + 2);
  int i_idx4 = i + n * (j + 3);

  int d_idx1 = dj + (65 * di);
  int d_idx2 = dj + (65 * (di + 1));
  int d_idx3 = dj + (65 * (di + 2));
  int d_idx4 = dj + (65 * (di + 3));

  int i1 = input[i_idx1];
  int i2 = input[i_idx2];
  int i3 = input[i_idx3];
  int i4 = input[i_idx4];

  data[d_idx1] = i1;
  data[d_idx2] = i2;
  data[d_idx3] = i3;
  data[d_idx4] = i4;

  __syncthreads();

  dj = 4 * threadIdx.y;
  int j0 = 64 * blockIdx.y;
  int i0 = 64 * blockIdx.x;
  int block_start = j0 + n * i0 + di;

  int d_idx1 = di + (65 * dj); 
  int d_idx2 = di + (65 * (dj + 1));
  int d_idx3 = di + (65 * (dj + 2));
  int d_idx4 = di + (65 * (dj + 3));

  int o_idx1 = block_start + n * dj;
  int o_idx2 = block_start + n * (dj + 1);
  int o_idx3 = block_start + n * (dj + 2);
  int o_idx4 = block_start + n * (dj + 3);

  int d1 = data[d_idx1];
  int d2 = data[d_idx2];
  int d3 = data[d_idx3];
  int d4 = data[d_idx4];

  output[o_idx1] = d1;
  output[o_idx2] = d2;
  output[o_idx3] = d3;
  output[o_idx4] = d4;
}

void cudaTranspose(const float *d_input,
                   float *d_output,
                   int n,
                   TransposeImplementation type) {
  if (type == NAIVE) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else if (type == SHMEM) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else if (type == OPTIMAL) {
    dim3 blockSize(64, 16);
    dim3 gridSize(n / 64, n / 64);
    optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
  } else {
    // unknown type
    assert(false);
  }
}
