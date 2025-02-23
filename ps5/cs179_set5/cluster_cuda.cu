#include <cassert>
#include <cuda_runtime.h>
#include "cluster_cuda.cuh"

// This assumes address stores the average of n elements atomically updates
// address to store the average of n + 1 elements (the n elements as well as
// val). This might be useful for updating cluster centers.
// modified from http://stackoverflow.com/a/17401122
__device__ 
float atomicUpdateAverage(float* address, int n, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i;
  int assumed;
  do {
    assumed = old;
    float next_val = (n * __int_as_float(assumed) + val) / (n + 1);
    old = ::atomicCAS(address_as_i, assumed,
		      __float_as_int(next_val));
  } while (assumed != old);
  return __int_as_float(old);
}


// computes the distance squared between vectors a and b where vectors have
// length size and stride stride.
__device__ 
float squared_distance(float *a, float *b, int stride, int size) {
  float dist = 0.0;
  for (int i=0; i < size; i++) {
    float diff = a[stride * i] - b[stride * i];
    dist += diff * diff;
  }
  return dist;
}

/*
 * Notationally, all matrices are column majors, so if I say that matrix Z is
 * of size m * n, then the stride in the m axis is 1. For purposes of
 * optimization (particularly coalesced accesses), you can change the format of
 * any array.
 *
 * clusters is a REVIEW_DIM * k array containing the location of each of the k
 * cluster centers.
 *
 * cluster_counts is a k element array containing how many data points are in 
 * each cluster.
 *
 * k is the number of clusters.
 *
 * data is a REVIEW_DIM * batch_size array containing the batch of reviews to
 * cluster. Note that each review is contiguous (so elements 0 through 49 are
 * review 0, ...)
 *
 * output is a batch_size array that contains the index of the cluster to which
 * each review is the closest to.
 *
 * batch_size is the number of reviews this kernel must handle.
 */
__global__
void sloppyClusterKernel(float *clusters, int *cluster_counts, int k, 
                          float *data, int *output, int batch_size) {
  extern __shared__ float smem[];

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  while (i < k * REVIEW_DIM) {
    smem[i] = clusters[i];
    i += gridDim.x * blockDim.x;
  }

  i = blockIdx.x * blockDim.x + threadIdx.x;

  while (i < batch_size) {
    float min_distance = squared_distance(smem, data, 1, REVIEW_DIM);
    int closest_cluster = 0;

    float *data_pt_start = data + REVIEW_DIM * i;

    // Iterate over the clusters, find the closest one.
    for (int j = 1; j < k; j++) {
      // store the distance between review i and cluster j
      float d = squared_distance(smem + REVIEW_DIM * j,
                                 data_pt_start, 1, REVIEW_DIM);
      if (d < min_distance) {
        min_distance = d;
        closest_cluster = j;
      }
    }

    // Update the cluster count of the cluster we are adding this data point to.
    output[i] = closest_cluster;
    int n = atomicAdd(cluster_counts + closest_cluster, 1);

    // Move the cluster location to the average of all in the cluster
    // after adding this point.
    for (int feature = 0; feature < REVIEW_DIM; feature++) {
      atomicUpdateAverage(clusters + REVIEW_DIM * closest_cluster, 
                          n, 
                          data_pt_start[feature]);
    }

    // Continue processing points left in the batch
    i += gridDim.x * blockDim.x;
  }
}


void cudaCluster(float *clusters, int *cluster_counts, int k,
		 float *data, int *output, int batch_size, 
		 cudaStream_t stream) {
  int block_size = (batch_size < 1024) ? batch_size : 1024;

  // grid_size = CEIL(batch_size / block_size)
  int grid_size = (batch_size + block_size - 1) / block_size;
  int shmem_bytes = sizeof(float) * k * REVIEW_DIM;

  sloppyClusterKernel<<<
    block_size, 
    grid_size, 
    shmem_bytes, 
    stream>>>(clusters, cluster_counts, k, data, output, batch_size);
}
