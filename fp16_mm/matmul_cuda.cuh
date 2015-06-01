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
#endif