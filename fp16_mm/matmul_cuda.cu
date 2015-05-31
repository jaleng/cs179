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