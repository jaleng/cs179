// matmul.cc
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>

#include "matmul_cuda.cuh"

// For the cublas example code
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
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


}
