// matmul.cc
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "matmul_cuda.cuh"

using half_float::half;

void print_half_matrix(half *h_p, int nrows, int ncols);
void print_single_matrix(float *fp, int nrows, int ncols);

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code,
                      const char *file,
                      int line,
                      bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}



int main(int argc, char *argv[]) {
  cudaEvent_t start;
  cudaEvent_t stop;
#define START_TIMER() {                         \
      gpuErrChk(cudaEventCreate(&start));       \
      gpuErrChk(cudaEventCreate(&stop));        \
      gpuErrChk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrChk(cudaEventRecord(stop));                     \
      gpuErrChk(cudaEventSynchronize(stop));                \
      gpuErrChk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrChk(cudaEventDestroy(start));                   \
      gpuErrChk(cudaEventDestroy(stop));                    \
    }



  // TESTING CODE

  // TODO: create 3 flp16 matrices
  int test_size = 128;
  int rows_a = test_size;
  int cols_a = test_size;
  int rows_b = test_size;
  int cols_b = test_size;
  int rows_c = test_size;
  int cols_c = test_size;

  // Create an identity matrix (array of fp16's)
  half *id = new half[test_size*test_size];

  for (int i = 0; i < test_size*test_size; ++i) {
    id[i] = half(0.0); 
  }
  
  for (int i = 0; i < test_size; ++i) {
    id[IDX2C(i, i, test_size)] = half(1.0);
  }
  
  // Create a matrix where the element is the column-major index
  // (array of fp16's)
  // Note that the values won't be exact as it is a fp16 approximation
  half *seq = new half[test_size*test_size];
  for (int i = 0; i < test_size*test_size; ++i) {
    seq[i] = half(i);
  }

  // Create float*'s that point to the beginning of the fp16-arrays
  float *id_half_fp = (float *) id;
  float *seq_half_fp = (float *) seq;


  //// DEBUG: try printing out the matrices we created
  //printf("\n\nPrinting out id.\n");
  //print_half_matrix(id, test_size, test_size);

  //printf("\n\nPrinting out seq.\n");
  //print_half_matrix(seq, test_size, test_size);

  // Assign half-array pointers for A and B, the matrices to be multiplied.
  half *a_hp = id;
  half *b_hp = seq;


  // Prepare the float-arrays for A and B to be used by cublas mm.

  // Host A float array
  float *h_A_farray = new float[test_size*test_size];
  for (int c = 0; c < cols_a; ++c) {
    for (int r = 0; r < rows_a; ++r) {
      h_A_farray[IDX2C(r, c, rows_a)] = a_hp[IDX2C(r, c, rows_a)];
    }
  }

  //// DEBUG: print host A float array to check for correctness
  //printf("***Printing host A float array. Should be id.\n");
  //print_single_matrix(h_A_farray, rows_a, cols_a);

  // Host B float array
  float *h_B_farray = new float[test_size*test_size];
  for (int c = 0; c < cols_b; ++c) {
    for (int r = 0; r < rows_b; ++r) {
      h_B_farray[IDX2C(r, c, rows_b)] = b_hp[IDX2C(r, c, rows_b)];
    }
  }

  //// DEBUG: print host B float array to check for correctness
  //printf("***Printing host B float array. Should be seq.\n");
  //print_single_matrix(h_B_farray, rows_b, cols_b);


  // Organize data on host.
  float *h_A_harray = id_half_fp;
  size_t A_sz_harray = (rows_a/2) * cols_a * sizeof(float);
  size_t A_sz_farray = A_sz_harray * 2;

  float *h_B_harray = seq_half_fp;
  size_t B_sz_harray = (rows_b/2) * cols_b * sizeof(float);
  size_t B_sz_farray = B_sz_harray * 2;

  // Allocate host memory for outputs for my kernel and cublas.
  float *h_C_mymmul = new float[(rows_a/2) * cols_b];
  float *h_C_cublas = new float[rows_a * cols_b];
  size_t C_sz_harray = (rows_a/2) * cols_b * sizeof(float);
  size_t C_sz_farray = C_sz_harray * 2;

  // Device pointers
  float *d_A_h;
  float *d_B_h;
  float *d_C_h;

  float *d_A_f;
  float *d_B_f;
  float *d_C_f;


  // Allocate memory on device
  gpuErrChk(cudaMalloc(&d_A_h, A_sz_harray));
  gpuErrChk(cudaMalloc(&d_B_h, B_sz_harray));
  gpuErrChk(cudaMalloc(&d_C_h, C_sz_harray));

  gpuErrChk(cudaMalloc(&d_A_f, A_sz_farray));
  gpuErrChk(cudaMalloc(&d_B_f, B_sz_farray));
  gpuErrChk(cudaMalloc(&d_C_f, C_sz_farray));

  // Copy A to device
  gpuErrChk(cudaMemcpy(d_A_h, h_A_harray, A_sz_harray, cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(d_A_f, h_A_farray, A_sz_farray, cudaMemcpyHostToDevice));

  // Copy B to device
  gpuErrChk(cudaMemcpy(d_B_h, h_B_harray, B_sz_harray, cudaMemcpyHostToDevice));
  gpuErrChk(cudaMemcpy(d_B_f, h_B_farray, B_sz_farray, cudaMemcpyHostToDevice));

  // DEBUG, clear out C
  gpuErrChk(cudaMemset(d_C_h, 0, C_sz_harray));
  gpuErrChk(cudaMemset(d_C_f, 0, C_sz_farray));


  /*****************************************************************************
   ***  MY KERNEL
   *****************************************************************************
   */
  // Run kernel, time it
  float my_kernel_time_ms = -1;
  START_TIMER();
  run_matmul_kernel(d_A_h, d_B_h, d_C_h, rows_a/2, cols_a, rows_b/2, cols_b);
  STOP_RECORD_TIMER(my_kernel_time_ms);
  printf("My kernel time was %fms.\n", my_kernel_time_ms);


  // Copy C from device to host
  gpuErrChk(cudaMemcpy(h_C_mymmul, d_C_h, C_sz_harray, cudaMemcpyDeviceToHost));


  /*****************************************************************************
   ***  CUBLAS
   *****************************************************************************
   */
  // Call the cublas initialization
  cublasStatus_t stat;
  cublasHandle_t handle;

  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
  }

  float alpha = 1.0;
  float beta = 0.0;
  float cublas_time_ms = -1;

  START_TIMER();
  stat = cublasSgemm(handle,
                     CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     test_size, test_size, test_size,
                     &alpha,
                     d_A_f, rows_a,
                     d_B_f, rows_b,
                     &beta,
                     d_C_f, rows_c
                     );
  STOP_RECORD_TIMER(cublas_time_ms);
  printf("CUBLAS time was %fms.\n", cublas_time_ms);


  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS Sgemm failed\n");
      return EXIT_FAILURE;
  }

  // Copy C from device to host
  gpuErrChk(cudaMemcpy(h_C_cublas, d_C_f, C_sz_farray, cudaMemcpyDeviceToHost));


  // Free A on device
  gpuErrChk(cudaFree(d_A_h));
  // Free B on device
  gpuErrChk(cudaFree(d_B_h));
  // Free C on device
  gpuErrChk(cudaFree(d_C_h));

  // Free A on device
  gpuErrChk(cudaFree(d_A_f));
  // Free B on device
  gpuErrChk(cudaFree(d_B_f));
  // Free C on device
  gpuErrChk(cudaFree(d_C_f));

  //// DEBUG: Print out the resulting matrices
  //half *h_C_hp = (half *) h_C_mymmul;
  //printf("\n\n*******Result from my kernel\n\n");
  //print_half_matrix(h_C_hp, test_size, test_size);
  //printf("\n\n*******Result from cublas kernel\n\n");
  //print_single_matrix(h_C_cublas, test_size, test_size);

  // Cleanup cublas
  cublasDestroy(handle);

  // Free host memory
  delete[] id;
  delete[] seq;
  delete[] h_C_mymmul;
  delete[] h_C_cublas;

}

void print_half_matrix(half *h_p, int nrows, int ncols) {
  printf("cols:    ");
  for (int c = 0; c < ncols; ++c) {
    printf("%15d ", c);
  }
  printf("\n");

  for (int r = 0; r < nrows; ++r) {
    printf("row %3d: ", r);
    for (int c = 0; c < ncols; ++c) {
      printf("%15f ", float(h_p[IDX2C(r, c, nrows)]));
    }
    printf("\n");
  }  
}

void print_single_matrix(float *fp, int nrows, int ncols) {
  printf("cols:    ");
  for (int c = 0; c < ncols; ++c) {
    printf("%15d ", c);
  }
  printf("\n");

  for (int r = 0; r < nrows; ++r) {
    printf("row %3d: ", r);
    for (int c = 0; c < ncols; ++c) {
      printf("%15f ", fp[IDX2C(r, c, nrows)]);
    }
    printf("\n");
  }  

}
