#ifndef CUDA_MATMUL_CUH
#define CUDA_MATMUL_CUH

#include "cublas_v2.h"
#include "half.hpp"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/*
void cudaDoSomething(const float *d_input,
                   float *d_output,
                   int n,
                   TransposeImplementation type);

*/
void run_matmul_kernel(float *a, float *b, float *c,
                       int rows_a, int cols_a, int rows_b, int cols_b);
#endif
