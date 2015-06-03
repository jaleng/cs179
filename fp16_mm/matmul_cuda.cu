// matmul_cuda.cu
#include <cassert>
#include <cuda_runtime.h>
#include "matmul_cuda.cuh"



/* EXAMPLE
__global__
void somethingKernel(...) {
  __shared__ float shmem[SIZE];
  // more code
}

// Call this from host
void cudaDoSomething(...) {
  dim3 blockSize(block_x, block_y);
  dim3 gridSize(grid_x, grid_y);
  somethingKernel<<<gridSize, blockSize>>>(...);
}
*/


/* Casting stuff
__device__ unsigned short __float2half_rn (float x)
__device__ float __half2float (unsigned short x)
__device__ float __int_as_float (int x)
__device__ int __float_as_int (float x)
*/
#define BLOCK_NCOLS 64
#define BLOCK_NROWS 32

__global__ void 
__launch_bounds__(1024, 2)
matmulKernel(float *a, float *b, float *c, int rows_a, int cols_a,
                  int rows_b, int cols_b) {
  __shared__ float shmem[32*64];

  float * shmem_A = shmem;
  //float * shmem_B = shmem + (32 * 64);

  // Right now we're assuming rows_a, cols_a, and cols_b are divisible by 32.
  int num_block_rows_in_c = rows_a / 32;
  int num_block_cols_in_c = cols_b / 64;

  int thread_row = threadIdx.x;
  int thread_col = threadIdx.y * 2;
  int block_row = blockIdx.x;
  int block_col = blockIdx.y;
  int num_block_cols = cols_a/64;

  int shmem_idx1 = thread_col * BLOCK_NROWS/2 + thread_row;
  int shmem_idx2 = (thread_col/2 + 32) * BLOCK_NROWS + thread_row;

  int offset1_for_a_to_shmem = IDX2C(thread_row, thread_col/2, rows_a);
  int offset2_for_a_to_shmem = IDX2C(thread_row, thread_col/2 + 32, rows_a);

  //int offset1_for_b_to_shmem = IDX2C(thread_row, thread_col/2 , rows_b);
  //int offset2_for_b_to_shmem = IDX2C(thread_row, thread_col/2 + 32, rows_b);

  int c_offset1 = IDX2C(thread_row, thread_col, rows_a);
  int c_offset2 = c_offset1 + rows_a;

  while (block_row < num_block_rows_in_c && block_col < num_block_cols_in_c) {

    int row_in_a = block_row * BLOCK_NROWS;
    int col_in_b = block_col * BLOCK_NCOLS;

    // Accumulators for this thread
    float acc11_i = 0;
    float acc12_i = 0;
    float acc21_i = 0;
    float acc22_i = 0;

    float acc11_ii = 0;
    float acc12_ii = 0;
    float acc21_ii = 0;
    float acc22_ii = 0;

    for (int block_col_idx = 0; 
             block_col_idx < num_block_cols; 
           ++block_col_idx) {
      // store A block (block_row, block_col_idx) into shmem
      //// A starts at a, and is column major
      //// We want to store the item in the thread_col column (of the block)
      ////                      and the thread_row row (of the block)

      int a_block_start_idx = IDX2C(row_in_a, 
                                    block_col_idx * BLOCK_NCOLS, 
                                    rows_a);
      int b_block_start_idx = IDX2C(block_col_idx * BLOCK_NROWS,
                                    col_in_b,
                                    rows_b);

      // Load the first 32 columns
      shmem_A[shmem_idx1] //DEBUG /2 on thread_col
        = a[a_block_start_idx + offset1_for_a_to_shmem];

      // Load the second 32 columns
      shmem_A[shmem_idx2]
        = a[a_block_start_idx + offset2_for_a_to_shmem];

      // store B block (block_col_idx, block_col) into shmem

      // Load the first 32 columns
      //shmem_B[shmem_idx1]
      //  = b[b_block_start_idx + offset1_for_b_to_shmem];

      // Load the second 32 columns
      //shmem_B[shmem_idx2]
      //  = b[b_block_start_idx + offset2_for_b_to_shmem];

      //// TRYING NEW THING
      // Read elem from column in B for later warp shuffling
      float b_col1_item_i = b[b_block_start_idx + IDX2C(thread_row, thread_col, rows_b)];
      //float b_col1_item_ii = b[b_block_start_idx + IDX2C(thread_row + 1, thread_col, rows_b)];
      float b_col2_item_i = b[b_block_start_idx + IDX2C(thread_row, thread_col + 1, rows_b)];
      //float b_col2_item_ii = b[b_block_start_idx + IDX2C(thread_row + 1, thread_col + 1, rows_b)];

      // Sync threads so shmem prepared for later accesses.
      __syncthreads();

      #pragma unroll
      for (int col_idx = 0; col_idx < BLOCK_NCOLS; col_idx += 4) {
        int thi1_i = IDX2C(thread_row, col_idx, BLOCK_NROWS);
        int thi1_ii = IDX2C(thread_row, col_idx + 2, BLOCK_NROWS);

        int thi2_i = IDX2C(thread_row, col_idx + 1, BLOCK_NROWS);
        int thi2_ii = IDX2C(thread_row, col_idx + 3, BLOCK_NROWS);

        // read 2 fp16's (1 float) from the a block (a11, a21)
        int two_halves1_i = __float_as_int(shmem_A[thi1_i]);

        // read 2 more (next col) from the a block (a12, a22)
        int two_halves2_i = __float_as_int(shmem_A[thi2_i]);

        //// Trying warp shuffle
        int two_halves3_i = __float_as_int(
                            __shfl(b_col1_item_i, col_idx/2));

        //int two_halves4 = __float_as_int(
        //               shmem_B[IDX2C(col_idx/2, thread_col + 1, BLOCK_NROWS)]);
        //// Trying warp shuffle
        int two_halves4_i = __float_as_int(
                            __shfl(b_col2_item_i, col_idx/2));

        // ii stuff
        int two_halves1_ii = __float_as_int(shmem_A[thi1_ii]);
        int two_halves2_ii = __float_as_int(shmem_A[thi2_ii]);
        int two_halves3_ii = __float_as_int(
                            __shfl(b_col1_item_i, col_idx/2  + 1));
        int two_halves4_ii = __float_as_int(
                            __shfl(b_col2_item_i, col_idx/2 + 1));


        unsigned short a11_i = (unsigned short) ((two_halves1_i << 16) >> 16);
        unsigned short a21_i = (unsigned short) (two_halves1_i >> 16);
        unsigned short a12_i = (unsigned short) ((two_halves2_i << 16) >> 16);
        unsigned short a22_i = (unsigned short) (two_halves2_i >> 16);

        unsigned short b21_i = (unsigned short) (two_halves3_i >> 16);
        unsigned short b11_i = (unsigned short) ((two_halves3_i << 16) >> 16);
        unsigned short b22_i = (unsigned short) (two_halves4_i >> 16);
        unsigned short b12_i = (unsigned short) ((two_halves4_i << 16) >> 16);

        //ii stuff
        unsigned short a11_ii = (unsigned short) ((two_halves1_ii << 16) >> 16);
        unsigned short a21_ii = (unsigned short) (two_halves1_ii >> 16);
        unsigned short a12_ii = (unsigned short) ((two_halves2_ii << 16) >> 16);
        unsigned short a22_ii = (unsigned short) (two_halves2_ii >> 16);

        unsigned short b21_ii = (unsigned short) (two_halves3_ii >> 16);
        unsigned short b11_ii = (unsigned short) ((two_halves3_ii << 16) >> 16);
        unsigned short b22_ii = (unsigned short) (two_halves4_ii >> 16);
        unsigned short b12_ii = (unsigned short) ((two_halves4_ii << 16) >> 16);

        float a11_f_i = __half2float(a11_i);
        float a21_f_i = __half2float(a21_i);
        float a12_f_i = __half2float(a12_i);
        float a22_f_i = __half2float(a22_i);

        float b11_f_i = __half2float(b11_i);
        float b21_f_i = __half2float(b21_i);
        float b12_f_i = __half2float(b12_i);
        float b22_f_i = __half2float(b22_i);

        float a11_f_ii = __half2float(a11_ii);
        float a21_f_ii = __half2float(a21_ii);
        float a12_f_ii = __half2float(a12_ii);
        float a22_f_ii = __half2float(a22_ii);

        float b11_f_ii = __half2float(b11_ii);
        float b21_f_ii = __half2float(b21_ii);
        float b12_f_ii = __half2float(b12_ii);
        float b22_f_ii = __half2float(b22_ii);

        acc11_i += a11_f_i * b11_f_i;
        acc21_i += a21_f_i * b11_f_i;
        acc12_i += a11_f_i * b12_f_i;
        acc22_i += a21_f_i * b12_f_i;

        acc11_ii += a11_f_ii * b11_f_ii;
        acc21_ii += a21_f_ii * b11_f_ii;
        acc12_ii += a11_f_ii * b12_f_ii;
        acc22_ii += a21_f_ii * b12_f_ii;

        acc11_i += a12_f_i * b21_f_i;
        acc21_i += a22_f_i * b21_f_i;
        acc12_i += a12_f_i * b22_f_i;
        acc22_i += a22_f_i * b22_f_i;

        acc11_ii += a12_f_ii * b21_f_ii;
        acc21_ii += a22_f_ii * b21_f_ii;
        acc12_ii += a12_f_ii * b22_f_ii;
        acc22_ii += a22_f_ii * b22_f_ii;
      }
    }

    // Prepare to store values
    unsigned short half11 = __float2half_rn(acc11_i + acc11_ii);
    unsigned short half21 = __float2half_rn(acc21_i + acc21_ii);

    unsigned short half12 = __float2half_rn(acc12_i + acc12_ii);
    unsigned short half22 = __float2half_rn(acc22_i + acc22_ii);

    float col_1_f = __int_as_float((((int) half21) << 16) | ((int) half11)); // DEBUG swapped half1 and 2
    float col_2_f = __int_as_float((((int) half22) << 16) | ((int) half12)); // DEBUG swapped half1 and 2

    // Store into the appropriate spot in C (in 1 write as a float)
    int c_block_start_idx = IDX2C(row_in_a, col_in_b, rows_a);


    // If the grid is not large enough to cover the entire matrix,
    // we must move to the next set of boxes to compute.
    // We do this in a column-major way.
    block_row += gridDim.x;
    if (block_row >= num_block_rows_in_c) {
      block_row = blockIdx.x;
      block_col += gridDim.y;
    }
    
    c[c_block_start_idx + c_offset1] = col_1_f;
    c[c_block_start_idx + c_offset2] = col_2_f;
  }
}

void run_matmul_kernel(float *a, float *b, float *c, 
                       int rows_a, int cols_a, int rows_b, int cols_b) {
  dim3 blockSize(32,32);
  dim3 gridSize(rows_a /32, cols_b / 64);

  matmulKernel<<<gridSize, blockSize>>>(a,b,c,rows_a, cols_a, rows_b, cols_b);
}
