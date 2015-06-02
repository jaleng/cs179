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

__global__
void matmulKernel(float *a, float *b, float *c, int rows_a, int cols_a,
                  int rows_b, int cols_b) {
  __shared__ float shmem[32*64*2];

  float * shmem_A = shmem;
  float * shmem_B = shmem + (32 * 64);

  // Right now we're assuming rows_a, cols_a, and cols_b are divisible by 32.
  int num_block_rows_in_c = rows_a / 32;
  int num_block_cols_in_c = cols_b / 64;

  // TODO(jg): make these constants
  int block_ncols = 64;
  int block_nrows = 32;

  int block_row = blockIdx.x;
  int block_col = blockIdx.y;
  int thread_row = threadIdx.x;
  int thread_col = threadIdx.y * 2; // DEBUG added * 2 try to fix prob where last 32 cols not written
  while (block_row < num_block_rows_in_c && block_col < num_block_cols_in_c) {
    int num_block_cols = cols_a/64; // Debug, added /64

    // Accumulators for this thread
    float acc11 = 0;
    float acc12 = 0;
    float acc21 = 0;
    float acc22 = 0;

    for (int block_col_idx = 0; block_col_idx < num_block_cols; ++block_col_idx) {
      // store A block (block_row, block_col_idx) into shmem
      //// A starts at a, and is column major
      //// We want to store the item in the thread_col column (of the block)
      ////                      and the thread_row row (of the block)

      // int a_block_start = block_col_idx * block_ncols * rows_a + block_row * block_nrows;
      // int b_block_start = block_col * block_ncols * rows_b + block_col_idx * block_nrows;

      int a_block_start_idx = IDX2C(block_row * block_nrows, 
                                    block_col_idx * block_ncols, 
                                    rows_a);
      int b_block_start_idx = IDX2C(block_col_idx * block_nrows,
                                    block_col * block_ncols,
                                    rows_b);

      // Load the first 32 columns
      shmem_A[IDX2C(thread_row, thread_col/2, block_nrows)] //DEBUG /2 on thread_col
        = a[a_block_start_idx + IDX2C(thread_row, thread_col/2, rows_a)];

      // Load the second 32 columns
      shmem_A[IDX2C(thread_row, thread_col/2 + 32, block_nrows)]
        = a[a_block_start_idx + IDX2C(thread_row, thread_col/2 + 32, rows_a)];

      // store B block (block_col_idx, block_col) into shmem

      // Load the first 32 columns
      shmem_B[IDX2C(thread_row, thread_col/2 , block_nrows)]
        = b[b_block_start_idx + IDX2C(thread_row, thread_col/2 , rows_b)];

      // Load the second 32 columns
      shmem_B[IDX2C(thread_row, thread_col/2 + 32, block_nrows)]
        = b[b_block_start_idx + IDX2C(thread_row, thread_col/2 + 32, rows_b)];

      // sync threads
      __syncthreads();


      for (int col_idx = 0; col_idx < block_ncols; col_idx += 2) {
        // read 2 fp16's (1 float) from the a block (a11, a21)
        int two_halves = __float_as_int(
                           shmem_A[IDX2C(thread_row, col_idx, block_nrows)]);
        unsigned short a21 = (unsigned short) (two_halves >> 16);
        unsigned short a11 = (unsigned short) ((two_halves << 16) >> 16);

        // read 2 more (next col) from the a block (a12, a22)
        two_halves = __float_as_int(
                       shmem_A[IDX2C(thread_row, col_idx + 1, block_nrows)]);
        unsigned short a22 = (unsigned short) (two_halves >> 16);
        unsigned short a12 = (unsigned short) ((two_halves << 16) >> 16);

        float a11_f = __half2float(a11);
        float a21_f = __half2float(a21);
        float a12_f = __half2float(a12);
        float a22_f = __half2float(a22);

        // read 2 fp16's (1 float) from the b block b1 (first row), b2 (next row)
        two_halves = __float_as_int(
                       shmem_B[IDX2C(col_idx/2, thread_col, block_nrows)]); // DEBUG put /2 after col_idx
        unsigned short b21 = (unsigned short) (two_halves >> 16);
        unsigned short b11 = (unsigned short) ((two_halves << 16) >> 16);

        two_halves = __float_as_int(
                       shmem_B[IDX2C(col_idx/2, thread_col + 1, block_nrows)]); // DEBUG put /2 after col_idx

        unsigned short b22 = (unsigned short) (two_halves >> 16);
        unsigned short b12 = (unsigned short) ((two_halves << 16) >> 16);

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
    unsigned short half1 = __float2half_rn(acc11);
    unsigned short half2 = __float2half_rn(acc21);
    float col_1_f = __int_as_float((((int) half2) << 16) | ((int) half1)); // DEBUG swapped half1 and 2

    half1 = __float2half_rn(acc12);
    half2 = __float2half_rn(acc22);
    float col_2_f = __int_as_float((((int) half2) << 16) | ((int) half1)); // DEBUG swapped half1 and 2

    // Store into the appropriate spot in C (in 1 write as a float)
    int c_block_start_idx = IDX2C(block_row * block_nrows, block_col * block_ncols, rows_a);
    c[c_block_start_idx + IDX2C(thread_row, thread_col, rows_a)] = col_1_f;
    c[c_block_start_idx + IDX2C(thread_row, thread_col + 1, rows_a)] = col_2_f;


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
