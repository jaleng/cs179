#include <cassert>
#include <cuda_runtime.h>
#include "transpose_cuda.cuh"

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
 * This kernel handles exactly 4 elements per thread. 
 * This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304  matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {

  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  for (; j < end_j; j++) {
    // Write output is non-coalesced, accesses are n floats apart,
    // so each will hit a new cache line. (32 cache lines touched).
    output[j + n * i] = input[i + n * j];
  }
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {

  __shared__ float data[4160]; // shmem array, 65x64
                               // padded to avoid bank conflicts

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  const int end_j = j + 4;

  int dj = 4 * threadIdx.y; // col idx within the 64x64 block
  int di = threadIdx.x;     // row idx within the 64x64 block

  for (; j < end_j; j++, dj++) {
    // Read from input coalesced, write to data in the transposed location.
    data[dj + (65 * di)] = input[i + n * j];
  }
  __syncthreads();

  j = 4 * threadIdx.y + 64 * blockIdx.y;
  dj = 4 * threadIdx.y;
  int j0 = 64 * blockIdx.y; // col offset of 64x64 block within the nxn array
  int i0 = 64 * blockIdx.x; // row offset of 64x64 block within the nxn array
  for (; j < end_j; j++, dj++) {
    // The block we are reading is already transposed, just need to
    // write it the transposed position of the block in the larger nxn array
    output[j0 + n * i0 + di + n * dj] = data[di + (65 * dj)];
  }


}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {

  __shared__ float data[4160]; // shmem array, 65x64
                               // padded to avoid bank conflicts

  const int i = threadIdx.x + 64 * blockIdx.x;
  int j = 4 * threadIdx.y + 64 * blockIdx.y;
  int dj = 4 * threadIdx.y; // col idx within the 64x64 block
  int di = threadIdx.x;     // row idx within the 64x64 block

  // Unrolled loop with separated index calculations
  // Arrangement of instructions altered to increase ILP

  int i_idx1 = i + n * j;
  int ditimes65 = 65 * di;
  int d_idx1 = dj + ditimes65;

  int i_idx2 = i_idx1 + n;
  int i_idx3 = i_idx1 + n + n;
  int i_idx4 = i_idx1 + n * 3;

  // Read input, coalesced
  float i1 = input[i_idx1];
  float i2 = input[i_idx2];
  float i3 = input[i_idx3];
  float i4 = input[i_idx4];

  int d_idx2 = d_idx1 + 1;
  int d_idx3 = d_idx1 + 2;
  int d_idx4 = d_idx1 + 3;

  // Write to shmem in transposed position, no bank conflicts
  data[d_idx1] = i1;
  data[d_idx2] = i2;
  data[d_idx3] = i3;
  data[d_idx4] = i4;

  int j0 = 64 * blockIdx.y;
  int i0 = 64 * blockIdx.x;
  int block_start = j0 + n * i0 + di;

  d_idx1 = di + (65 * dj); 
  int o_idx1 = block_start + n * dj;

  d_idx2 = d_idx1 + 65;
  d_idx3 = d_idx1 + 130;
  d_idx4 = d_idx1 + 195;

  int o_idx2 = o_idx1 + n;
  int o_idx3 = o_idx1 + n + n;
  int o_idx4 = o_idx1 + 3 * n;

  __syncthreads();

  // Read from shmem, no bank conflicts
  float d1 = data[d_idx1];
  float d2 = data[d_idx2];
  float d3 = data[d_idx3];
  float d4 = data[d_idx4];

  // Write to output, coalesced
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
