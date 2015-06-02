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

__global__
void matmulKernel(float *a, float *b, float *c, int rows_a, int cols_a,
                  int rows_b, int cols_b) {
  __shared__ float shmem[32*64];

  float * shmem_A = shmem;
  //float * shmem_B = shmem + (32 * 64);

  // Right now we're assuming rows_a, cols_a, and cols_b are divisible by 32.
  int num_block_rows_in_c = rows_a / 32;
  int num_block_cols_in_c = cols_b / 64;

  int block_row = blockIdx.x;
  int block_col = blockIdx.y;
  int thread_row = threadIdx.x;
  int thread_col = threadIdx.y * 2;
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

    // Accumulators for this thread
    float acc11 = 0;
    float acc12 = 0;
    float acc21 = 0;
    float acc22 = 0;

    int row_in_a = block_row * BLOCK_NROWS;
    int col_in_b = block_col * BLOCK_NCOLS;

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

      // Sync threads so shmem prepared for later accesses.
      __syncthreads();

      //// TRYING NEW THING
      // Read elem from column in B for later warp shuffling
      float b_col1_item = b[b_block_start_idx + IDX2C(thread_row, thread_col, rows_b)];
      float b_col2_item = b[b_block_start_idx + IDX2C(thread_row, thread_col + 1, rows_b)];



      for (int col_idx = 0; col_idx < BLOCK_NCOLS; col_idx += 2) {
        // read 2 fp16's (1 float) from the a block (a11, a21)
        int two_halves1 = __float_as_int(
                           shmem_A[IDX2C(thread_row, col_idx, BLOCK_NROWS)]);
        unsigned short a21 = (unsigned short) (two_halves1 >> 16);
        unsigned short a11 = (unsigned short) ((two_halves1 << 16) >> 16);

        // read 2 more (next col) from the a block (a12, a22)
        int two_halves2 = __float_as_int(
                       shmem_A[IDX2C(thread_row, col_idx + 1, BLOCK_NROWS)]);
        unsigned short a22 = (unsigned short) (two_halves2 >> 16);
        unsigned short a12 = (unsigned short) ((two_halves2 << 16) >> 16);

        float a11_f = __half2float(a11);
        float a21_f = __half2float(a21);
        float a12_f = __half2float(a12);
        float a22_f = __half2float(a22);


        // read 2 fp16's (1 float) from the b block b1 (first row), b2 (next row)
        //int two_halves3 = __float_as_int(
        //               shmem_B[IDX2C(col_idx/2, thread_col, BLOCK_NROWS)]);
        //// Trying warp shuffle
        int two_halves3 = __float_as_int(
                            __shfl(b_col1_item, col_idx/2));

        unsigned short b21 = (unsigned short) (two_halves3 >> 16);
        unsigned short b11 = (unsigned short) ((two_halves3 << 16) >> 16);

        //int two_halves4 = __float_as_int(
        //               shmem_B[IDX2C(col_idx/2, thread_col + 1, BLOCK_NROWS)]);
        //// Trying warp shuffle
        int two_halves4 = __float_as_int(
                            __shfl(b_col2_item, col_idx/2));

        unsigned short b22 = (unsigned short) (two_halves4 >> 16);
        unsigned short b12 = (unsigned short) ((two_halves4 << 16) >> 16);

        float b11_f = __half2float(b11);
        float b21_f = __half2float(b21);
        float b12_f = __half2float(b12);
        float b22_f = __half2float(b22);

        acc11 += a11_f * b11_f + a12_f * b21_f;
        acc21 += a21_f * b11_f + a22_f * b21_f;
        acc12 += a11_f * b12_f + a12_f * b22_f;
        acc22 += a21_f * b12_f + a22_f * b22_f;
      }
    }

    // Prepare to store values
    unsigned short half11 = __float2half_rn(acc11);
    unsigned short half21 = __float2half_rn(acc21);
    float col_1_f = __int_as_float((((int) half21) << 16) | ((int) half11)); // DEBUG swapped half1 and 2

    unsigned short half12 = __float2half_rn(acc12);
    unsigned short half22 = __float2half_rn(acc22);
    float col_2_f = __int_as_float((((int) half22) << 16) | ((int) half12)); // DEBUG swapped half1 and 2

    // Store into the appropriate spot in C (in 1 write as a float)
    int c_block_start_idx = IDX2C(block_row * BLOCK_NROWS, block_col * BLOCK_NCOLS, rows_a);
    c[c_block_start_idx + c_offset1] = col_1_f;
    c[c_block_start_idx + c_offset2] = col_2_f;


    // If the grid is not large enough to cover the entire matrix,
    // we must move to the next set of boxes to compute.
    // We do this in a column-major way.
    block_row += gridDim.x;
    if (block_row >= num_block_rows_in_c) {
      block_row = blockIdx.x;
      block_col += gridDim.y;
    }
  }
}

void run_matmul_kernel(float *a, float *b, float *c, 
                       int rows_a, int cols_a, int rows_b, int cols_b) {
  dim3 blockSize(32,32);
  dim3 gridSize(rows_a /32, cols_b / 64);

  matmulKernel<<<gridSize, blockSize>>>(a,b,c,rows_a, cols_a, rows_b, cols_b);
}
