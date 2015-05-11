#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "cluster_cuda.cuh"

using namespace std;

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

// timing setup code
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

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) {
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0.0, 1.0);
  for (int i=0; i < size; i++) {
    output[i] = distribution(generator);
  }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM floats.
void readLSAReview(string review_str, float *output) {
  stringstream stream(review_str);
  int component_idx = 0;

  for (string component; getline(stream, component, ','); component_idx++) {
    output[component_idx] = atof(component.c_str());
  }
  assert(component_idx == REVIEW_DIM);
}

// used to pass arguments to printerCallback
struct printerArg {
  int review_idx_start;
  int batch_size;
  int *cluster_assignments;
};

// Prints out which cluster each review in a batch was assigned to.
// TODO: Call with cudaStreamAddCallback (after completing D->H memcpy)
void printerCallback(cudaStream_t stream, cudaError_t status, void *userData) {
  printerArg *arg = static_cast<printerArg *>(userData);

  for (int i=0; i < arg->batch_size; i++) {
    printf("%d: %d\n", 
	   arg->review_idx_start + i, 
	   arg->cluster_assignments[i]);
  }

  delete arg;
}

void cluster(istream& in_stream, int k, int batch_size) {
  // TODO(jg) remove
  printf("In cluster()\n");

  // cluster centers
  float *d_clusters;

  // how many points lie in each cluster
  int *d_cluster_counts;

  // allocate memory for cluster centers and counts
  gpuErrChk(cudaMalloc(&d_clusters, k * REVIEW_DIM * sizeof(float)));
  gpuErrChk(cudaMalloc(&d_cluster_counts, k * sizeof(int)));

  // randomly initialize cluster centers
  float *clusters = new float[k * REVIEW_DIM];
  gaussianFill(clusters, k * REVIEW_DIM);
  gpuErrChk(cudaMemcpy(d_clusters, clusters, k * REVIEW_DIM * sizeof(float),
		       cudaMemcpyHostToDevice));

  // initialize cluster counts to 0
  gpuErrChk(cudaMemset(d_cluster_counts, 0, k * sizeof(int)));
  
  // TODO: allocate copy buffers and streams
  // TODO(jg): create 2 streams
  // TODO(jg) remove
  printf("Creating streams\n");

  cudaStream_t streams[2];
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);

  // TODO(jg): create 2 pinned host buffers using cudaMallocHost
  // TODO(jg) remove
  printf("Creating hostBuffers\n");

  float* hostBuffers[2];
  gpuErrChk(
    cudaMallocHost(&hostBuffers[0], sizeof(float) * REVIEW_DIM * batch_size) );
  gpuErrChk(
    cudaMallocHost(&hostBuffers[1], sizeof(float) * REVIEW_DIM * batch_size) );

  // TODO(jg) remove
  printf("Creating hostOut\n");

  int* hostOut[2];
  gpuErrChk(
    cudaMallocHost(&hostOut[0], sizeof(int) *  batch_size) );
  gpuErrChk(
    cudaMallocHost(&hostOut[1], sizeof(int) *  batch_size) );

  // TODO(jg): create 2 buffers on GPU
  // TODO(jg) remove
  printf("Creating devBuffers\n");

  float* devBuffers[2];
  gpuErrChk(
    cudaMalloc(&devBuffers[0], sizeof(float) * REVIEW_DIM * batch_size) );
  gpuErrChk(
    cudaMalloc(&devBuffers[1], sizeof(float) * REVIEW_DIM * batch_size) );

  // TODO(jg) remove
  printf("Creating devOut\n");

  int* devOut[2];
  gpuErrChk(
    cudaMalloc(&devOut[0], sizeof(int) *  batch_size) );
  gpuErrChk(
    cudaMalloc(&devOut[1], sizeof(int) *  batch_size) );

  // TODO(jg) remove
  printf("Starting main loop to process input lines\n");

  // main loop to process input lines (each line corresponds to a review)
  int review_idx = 0;
  for (string review_str; getline(in_stream, review_str); review_idx++) {
    int i = (review_idx / batch_size) % 2;
    int j = review_idx % batch_size;

    // TODO: readLSAReview into appropriate storage
    readLSAReview(review_str, devBuffers[i] + j * REVIEW_DIM);

    // TODO: if you have filled up a batch, copy H->D, kernel, copy D->H,
    //       and set callback to printerCallback. Will need to allocate
    //       printerArg struct. Do all of this in a stream.
    if (j == batch_size - 1) {
      cudaMemcpyAsync(devBuffers[i], hostBuffers[i], 
                      sizeof(float) * REVIEW_DIM * batch_size,
                      cudaMemcpyHostToDevice,
                      streams[i]);
      cudaCluster(d_clusters, d_cluster_counts, k, devBuffers[i], devOut[i],
                  batch_size, streams[i]);
      cudaMemcpyAsync(hostOut[i], devOut[i], sizeof(int) * batch_size,
                      cudaMemcpyDeviceToHost,
                      streams[i]);
      printerArg *arg = new printerArg;
      arg->review_idx_start = review_idx - j;
      arg->batch_size = batch_size;
      arg->cluster_assignments = hostOut[i];

      cudaStreamAddCallback(streams[i], printerCallback, (void*) arg, 0);

    }
  }

  // wait for everything to end on GPU before final summary
  gpuErrChk(cudaDeviceSynchronize());

  // TODO(jg) remove
  printf("Finished device synchronize after main loop\n");

  // retrieve final cluster locations and counts
  int *cluster_counts = new int[k];
  gpuErrChk(cudaMemcpy(cluster_counts, d_cluster_counts, k * sizeof(int), 
		       cudaMemcpyDeviceToHost));
  gpuErrChk(cudaMemcpy(clusters, d_clusters, k * REVIEW_DIM * sizeof(int),
		       cudaMemcpyDeviceToHost));

  // print cluster summaries
  for (int i=0; i < k; i++) {
    printf("Cluster %d, population %d\n", i, cluster_counts[i]);
    printf("[");
    for (int j=0; j < REVIEW_DIM; j++) {
      printf("%.4e,", clusters[i * REVIEW_DIM + j]);
    }
    printf("]\n\n");
  }

  // free cluster data
  gpuErrChk(cudaFree(d_clusters));
  gpuErrChk(cudaFree(d_cluster_counts));
  delete[] cluster_counts;
  delete[] clusters;

  // TODO: finish freeing memory, destroy streams
}

int main(int argc, char** argv) {
  int k = 5;
  int batch_size = 32;

  if (argc == 1) {
    cluster(cin, k, batch_size);
  } else if (argc == 2) {
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    cluster(buffer, k, batch_size);
  }
}
