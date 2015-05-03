
/* 
Based off work by Nelson, et al.
Brigham Young University (2010)

Adapted by Kevin Yuh (2015)
*/


#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>

#define PI 3.14159265358979


/* Check errors on CUDA runtime functions */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}



/* Check errors on cuFFT functions */
void gpuFFTchk(int errval){
    if (errval != CUFFT_SUCCESS){
        printf("Failed FFT call, error code %d\n", errval);
    }
}


/* Check errors on CUDA kernel calls */
void checkCUDAKernelError()
{
    cudaError_t err = cudaGetLastError();
    if  (cudaSuccess != err){
        fprintf(stderr, "Error %s\n", cudaGetErrorString(err));
    } else {
        fprintf(stderr, "No kernel error detected\n");
    }

}

// TODO(jg) comment
__global__
void cudaRampFilterKernel(cufftComplex *input, int width, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < width * n) {
        int j = i % width;
        j = width / 2 - abs(width / 2 - j);
        float scale = float(j) / float(width / 2);
        cufftComplex val = input[i];
        val.x *= scale;
        val.y *= scale;
        input[i] = scale;

        // Move to next set of blocks to be processed
        i += gridDim.x * blockDim.x;
    }
}


// TODO(jg) comment
__global__
void cudaExtractRealKernel(cufftComplex *input, float *output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < size) {
        output[i] = input[i].x;

        // Move to next set of blocks to be processed
        i += gridDim.x * blockDim.x;
    }
}

// TODO(jg) comment
__global__
void cudaCTBackProjection(
    texture<float, 2, cudaReadModeElementType> texreference,
    float *sinogram;
    int sinogram_width;
    float *output,
    int height,
    int width,
    int nAngles) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (i < height * width * nAngles) {
        int y = i / (width * nAngles);
        int x = (i % (width * nAngles)) / nAngles;
        int theta_idx = i % nAngles;
        float theta = (PI / nAngles) * theta_idx;

        float x_0 = x - width / 2;
        float y_0 = height / 2 - y;
        float d;

        if (theta_idx == 0) {
            d = x_0;
        } else if (theta_idx == (nAngles / 2) && (nAngles % 2 == 1)) {
            d = y_0;
        } else {
            // Calculate m from theta
            float m = -cos(theta) / sin(theta);
            float q = -1/m;
            
            float x_i = (y_0 - m * x_0) / (q - m);
            float y_i = q * x_i;
            d = sqrt(x_i * x_i + y_i * y_i);
            if (x_i < 0 || (q < 0 && x_i > 0)) {
                d = -d;
            }
        }

        d_idx = sinogram_width / 2 + d;
        output[y * width + x] += tex2D(texreference, theta_idx, d_idx);
        // Calculate x_i, y_i from m, -1/m
        // Calculate d from x_i, y_i
        // image[x,y] += sinogram[theta, "distance"]
        // Move to next set of blocks to be processed
        i += gridDim.x * blockDim.x;
    }    
}


int main(int argc, char** argv){

    if (argc != 7){
        fprintf(stderr, "Incorrect number of arguments.\n\n");
        fprintf(stderr, "\nArguments: \n \
        < Sinogram filename > \n \
        < Width or height of original image, whichever is larger > \n \
        < Number of angles in sinogram >\n \
        < threads per block >\n \
        < number of blocks >\n \
        < output filename >\n");
        exit(EXIT_FAILURE);
    }


    /********** Parameters **********/

    int width = atoi(argv[2]);
    int height = width;
    int sinogram_width = (int)ceilf( height * sqrt(2) );

    int nAngles = atoi(argv[3]);


    int threadsPerBlock = atoi(argv[4]);
    int nBlocks = atoi(argv[5]);


    /********** Data storage *********/


    // GPU DATA STORAGE
    cufftComplex *dev_sinogram_cmplx;
    float *dev_sinogram_float; 
    float* output_dev;  // Image storage


    cufftComplex *sinogram_host;

    size_t size_result = width*height*sizeof(float);
    float *output_host = (float *)malloc(size_result);




    /*********** Set up IO, Read in data ************/

    sinogram_host = (cufftComplex *)malloc(  sinogram_width*nAngles*sizeof(cufftComplex) );

    FILE *dataFile = fopen(argv[1],"r");
    if (dataFile == NULL){
        fprintf(stderr, "Sinogram file missing\n");
        exit(EXIT_FAILURE);
    }

    FILE *outputFile = fopen(argv[6], "w");
    if (outputFile == NULL){
        fprintf(stderr, "Output file cannot be written\n");
        exit(EXIT_FAILURE);
    }

    int j, i;

    for(i = 0; i < nAngles * sinogram_width; i++){
        fscanf(dataFile,"%f",&sinogram_host[i].x);
        sinogram_host[i].y = 0;
    }

    fclose(dataFile);


    /*********** Assignment starts here *********/

    /* TODO: Allocate memory for all GPU storage above, copy input sinogram
    over to dev_sinogram_cmplx. */

    // Allocate device memory for the sinogram complex values
    gpuErrchk(cudaMalloc( (void**) &dev_sinogram_cmplx,
                          sinogram_width*nAngles*sizeof(cufftComplex) ));

    // TODO(jg): is this size right?
    // Allocate device memory for the sinogram real values
    gpuErrchk(cudaMalloc( (void**) &dev_sinogram_float,
                          sinogram_width*nAngles*sizeof(float) ));

    // Allocate device memory for output
    gpuErrchk(cudaMalloc( (void**) &output_dev,
                          size_result ));
    gpuErrchk(cudaMemset(output_dev, 0, size_result));

    // Copy the sinogram from host to device
    gpuErrchk(cudaMemcpy( dev_sinogram_cmplx,
                          sinogram_host,
                          sinogram_width*nAngles*sizeof(cufftComplex),
                          cudaMemcpyHostToDevice ));



    /* TODO 1: Implement the high-pass filter:
        - Use cuFFT for the forward FFT
        - Create your own kernel for the frequency scaling.
        - Use cuFFT for the inverse FFT
        - extract real components to floats
        - Free the original sinogram (dev_sinogram_cmplx)

        Note: If you want to deal with real-to-complex and complex-to-real
        transforms in cuFFT, you'll have to slightly change our code above.
    */

    // Create a cuFFT plan to FFT the sinogram
    cufftHandle plan;
    cufftPlan1d(&plan, sinogram_width, CUFFT_C2C, nAngles);

    // Run forward fft on sinogram
    gpuFFTchk(cufftExecC2C(plan, dev_sinogram_cmplx, 
                           dev_sinogram_cmplx, CUFFT_FORWARD));

    // TODO(jg): frequency scaling
    checkCUDAKernelError(
        cudaRampFilterKernel<<<512, 200>>>( dev_sinogram_cmplx,
                                            sinogram_width,
                                            nAngles ));
    // TODO(jg): inverse fft on sinogram
    gpuFFTchk(cufftExecC2C( plan, dev_sinogram_cmplx,
                            dev_sinogram_cmplx, CUFFT_INVERSE ));

    // TODO(jg): Extract reals
    checkCUDAKernelError(
        cudaExtractRealKernel<<<512, 200>>>(
            dev_sinogram_cmplx, dev_sinogram_float, sinogram_width*nAngles) );

    // TODO(jg): Free the original sinogram
    free(dev_sinogram_cmplx);

    /* TODO 2: Implement backprojection.
        - Allocate memory for the output image.
        - Create your own kernel to accelerate backprojection.
        - Copy the reconstructed image back to output_host.
        - Free all remaining memory on the GPU.
    */
    
    // Set up 2D texture cache on sinogram
    texture<float, 2, cudaReadModeElementType> texreference;
    cudaArray *cSinogram;
    cudaChannelFormatDesc channel = cudaCreateChannelDesc<float>();

    cudaMallocArray(&cSinogram, &channel, sinogram_width, nAngles);

    cudaMemcpyToArray(cSinogram,
                      0,
                      0,
                      dev_sinogram_float, 
                      sinogram_width*nAngles*sizeof(float), 
                      cudaMemcpyDeviceToDevice);
    texreference.filterMode = cudaFilterModeLinear;
    texreference.addressMode[0] = cudaAddressModeClamp;
    texreference.addressMode[1] = cudaAddressModeClamp;

    cudaBindTextureToArray(texreference, cSinogram);

    // TODO(jg): Call backproject kernel
    checkCUDAKernelError(
        cudaCTBackProjection<<<512, 200>>>(
            texreference,
            dev_sinogram_float,
            sinogram_width,
            output_dev,
            height,
            width,
            nAngles) );

    //
    cudaUnbindTexture(texreference);

    // Copy result from device to host
    gpuErrchk(cudaMemcpy(output_host,
                         output_dev,
                         sinogram_width*nAngles*sizeof(float), 
                         cudaMemcpyDeviceToHost));

    cudaFreeArray(cSinogram);
    
    /* Export image data. */

    for(j = 0; j < width; j++){
        for(i = 0; i < height; i++){
            fprintf(outputFile, "%e ",output_host[j*width + i]);
        }
        fprintf(outputFile, "\n");
    }


    /* Cleanup: Free host memory, close files. */

    free(sinogram_host);
    free(output_host);

    fclose(outputFile);

    return 0;
}




