// matmul.cc
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "matmul_cuda.cuh"

using half_float::half;

print_half_matrix(half *h_p, int nrows, int ncols);

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


// fills fill with random numbers is [0, 1]. Size is number of elements to
// assign
void randomFill(float *fill, int size) {
  for (int i = 0; i < size; i++) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    fill[i] = r;
  }
}

// For the cublas example code
static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
    cublasSscal (handle, n-p, &alpha, &m[IDX2C(p,q,ldm)], ldm);
    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
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

  /** Timing example
   *
   * // initialize timer
   * float my_time_ms = -1;
   * START_TIMER();
   * // Do thing to be timed
   * STOP_RECORD_TIMER(my_time_ms);
   * printf("My time was: %f ms \n", my_time_ms);
   * END Timing example 
   */

  // Some cublas example code to get started
  // START CUBLAS EXAMPLE CODE
  /*********************************************************
  float cublas_ex_code_time_ms = -1;
  START_TIMER();
#define M 6
#define N 5
  cudaError_t cudaStat;    
  cublasStatus_t stat;
  cublasHandle_t handle;
  int i, j;
  float* devPtrA;
  float* a = 0;
  a = (float *)malloc (M * N * sizeof (*a));
  if (!a) {
      printf ("host memory allocation failed");
      return EXIT_FAILURE;
  }
  for (j = 0; j < N; j++) {
      for (i = 0; i < M; i++) {
          a[IDX2C(i,j,M)] = (float)(i * M + j + 1);
      }
  }
  cudaStat = cudaMalloc ((void**)&devPtrA, M*N*sizeof(*a));
  if (cudaStat != cudaSuccess) {
      printf ("device memory allocation failed");
      return EXIT_FAILURE;
  }
  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("CUBLAS initialization failed\n");
      return EXIT_FAILURE;
  }
  stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data download failed");
      cudaFree (devPtrA);
      cublasDestroy(handle);
      return EXIT_FAILURE;
  }
  modify (handle, devPtrA, M, N, 1, 2, 16.0f, 12.0f);
  stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
      printf ("data upload failed");
      cudaFree (devPtrA);
      cublasDestroy(handle);
      return EXIT_FAILURE;
  }
  cudaFree (devPtrA);
  cublasDestroy(handle);
  for (j = 0; j < N; j++) {
      for (i = 0; i < M; i++) {
          printf ("%7.0f", a[IDX2C(i,j,M)]);
      }
      printf ("\n");
  }
  free(a);
  // return EXIT_SUCCESS;
  // END CUBLAS EXAMPLE CODE
  STOP_RECORD_TIMER(cublas_ex_code_time_ms);
  printf("My time was: %f ms \n", cublas_ex_code_time_ms);

  *****************************************************************************/

  // TESTING CODE

  // TODO: create 3 64x64 flp16 matrix
  int rows_a = 64;
  int cols_a = 64;
  int rows_b = 64;
  int cols_b = 64;
  int rows_c = 64;
  int cols_c = 64;

  half *id_64x64 = new half[64*64];

  for (int i = 0; i < 64*64; ++i) {
    id_64x64[i] = half(1.0); // DEBUG, was 0.0
  }
  /* DEBUG, was uncommented
  for (int i = 0; i < 64; ++i) {
    id_64x64[IDX2C(i, i, 64)] = half(1.0);
  }
  */

  half *seq_64x64 = new half[64*64];
  for (int i = 0; i < 64*64; ++i) {
    seq_64x64[i] = half(1.0); // DEBUG, was half(i);
  }

  float *id_64x64_fp = (float *) id_64x64;
  float *seq_64x64_fp = (float *) seq_64x64;


  // DEBUG: try printing out the matrices we created
  printf("\n\nPrinting out id_64x64.\n");
  print_half_matrix(id_64x64, 64, 64);

  printf("\n\nPrinting out seq_64x64.\n");
  print_half_matrix(seq_64x64, 64, 64);




  // Start timing our multiplication
  float my_kernal_time_ms = -1;
  START_TIMER();

  // Allocate memory for A on device
  // Allocate memory for B on device
  // Allocate memory for C on device
  float *h_A = id_64x64_fp;
  size_t A_sz = (rows_a/2) * cols_a * sizeof(float);

  float *h_B = seq_64x64_fp;
  size_t B_sz = (rows_b/2) * cols_b * sizeof(float);

  float *h_C = new float[(rows_a/2) * cols_b];
  size_t C_sz = (rows_a/2) * cols_b * sizeof(float);

  float *d_A;
  float *d_B;
  float *d_C;

  gpuErrChk(cudaMalloc(&d_A, A_sz));
  gpuErrChk(cudaMalloc(&d_B, B_sz));
  gpuErrChk(cudaMalloc(&d_C, C_sz));

  // Copy A to device
  gpuErrChk(cudaMemcpy(d_A, h_A, A_sz, cudaMemcpyHostToDevice));

  // Copy B to device
  gpuErrChk(cudaMemcpy(d_B, h_B, B_sz, cudaMemcpyHostToDevice));

  // DEBUG, clear out C
  gpuErrChk(cudaMemset(d_C, 1, C_sz));

  // Run kernel
  run_matmul_kernel(d_A, d_B, d_C, rows_a/2, cols_a, rows_b/2, cols_b);

  // Copy C from device to host
  gpuErrChk(cudaMemcpy(h_C, d_C, C_sz, cudaMemcpyDeviceToHost));

  // Free A on device
  gpuErrChk(cudaFree(d_A));
  // Free B on device
  gpuErrChk(cudaFree(d_B));
  // Free C on device
  gpuErrChk(cudaFree(d_C));

  STOP_RECORD_TIMER(my_kernal_time_ms);

  printf("My kernel time was %fms.\n", my_kernal_time_ms);

  // Print out C -- convert halfs to floats and print those
  half *h_C_hp = (half *) h_C;
  // DEBUG, was uncommented
  //print_half_matrix(h_C_hp, 64, 64);

  // Free host memory
  delete[] id_64x64;
  delete[] seq_64x64;
  delete[] h_C;

}

print_half_matrix(half *h_p, int nrows, int ncols) {
  printf("cols:    ");
  for (int c = 0; c < ncols; ++c) {
    printf("%15d ", c);
  }
  printf("\n");

  for (int r = 0; r < nrows; ++r) {
    printf("row %3d: ", r);
    for (int c = 0; c < cols_c; ++c) {
      printf("%15f ", float(h_p[IDX2C(r, c, nrows)]));
    }
    printf("\n");
  }  
}