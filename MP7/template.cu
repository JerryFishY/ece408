// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define SCAN_BLOCK_SIZE 128
#define BLOCK_SIZE 512
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


//@@ insert code here
// One thread is in charge of several pixels
__global__ void greyscale(float* input, float* globalhisto, int greysize){
  int tx = threadIdx.x;
  int gx = blockIdx.x*blockDim.x + tx; // the total number of pixels
  __shared__ int histogram[HISTOGRAM_LENGTH];
  if(tx<HISTOGRAM_LENGTH){
    histogram[tx] = 0;
  }
  __syncthreads();
  int stride = blockDim.x*gridDim.x;
  while(gx < greysize){
    // load the image into shared memory
    float r = input[gx*3];
    float g = input[gx*3+1];
    float b = input[gx*3+2];

    unsigned char grey = (unsigned char) (0.299 * (unsigned char) (255 * r) + 0.587*(unsigned char) (255 * g) + 0.114*(unsigned char) (255 * b));
    atomicAdd(&histogram[grey],1);
    gx += stride;
  }
  __syncthreads(); 
  if(tx < HISTOGRAM_LENGTH){
    atomicAdd(&globalhisto[tx], histogram[tx]);
  }
}

__global__ void scan(float *histogram, float* cdf, int greysize) {
  int tx = threadIdx.x;
  int gx = blockIdx.x*blockDim.x*2 + tx;
  __shared__ float SharedInput[2*SCAN_BLOCK_SIZE];
  // load data into shared memory
  if(gx < HISTOGRAM_LENGTH) {
    SharedInput[tx] = histogram[gx]/greysize;
  } else {
    SharedInput[tx] = 0;
  }
  if(gx + blockDim.x < HISTOGRAM_LENGTH) {
    SharedInput[tx + blockDim.x] = histogram[gx + blockDim.x]/greysize;
  } else {
    SharedInput[tx + blockDim.x] = 0;
  }
  __syncthreads();

  // reduction
  for (size_t stride = 1;stride<=SCAN_BLOCK_SIZE;stride*=2) {
    int index = (tx+1)*stride*2-1;
    if(index < 2*SCAN_BLOCK_SIZE) {
      SharedInput[index] = SharedInput[index-stride]+SharedInput[index];
    }
    __syncthreads();
  }
  // distribution
  for (size_t stride = SCAN_BLOCK_SIZE / 2;stride > 0; stride /= 2) {
    int index = (tx+1)*stride*2-1;
    if(index+stride < 2*SCAN_BLOCK_SIZE) {
      SharedInput[index+stride] += SharedInput[index];
    }
    __syncthreads();
  }
    // Wirte back to global
    if (gx < HISTOGRAM_LENGTH)
    {
      cdf[gx] = SharedInput[tx];
    }
  if(gx + blockDim.x < HISTOGRAM_LENGTH) {
    cdf[gx + blockDim.x] = SharedInput[tx + blockDim.x];
  }
}

__global__ void recolor(float* input, float* output, float* cdf, int imagesize){
  // First load the cdf into shared memory
  __shared__ float sharedcdf[HISTOGRAM_LENGTH];
  int tx = threadIdx.x;
  int gx = blockIdx.x*blockDim.x + tx;
  if(tx < HISTOGRAM_LENGTH) {
    sharedcdf[tx] = cdf[tx];
  }
  __syncthreads();
  // Then recolor the image using the cdf
  while(gx < imagesize) {
    unsigned char color = (unsigned char)(input[gx]*255);
    float color1 = (sharedcdf[color] - sharedcdf[0]) / (1.0 - sharedcdf[0]);
    output[gx] = color1;
    gx += blockDim.x*gridDim.x;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceGreyImage;
  float *deviceHistogram;
  float *deviceCDF;
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  size_t imagesize = imageWidth * imageHeight * imageChannels;
  size_t greysize = imageWidth * imageHeight;
  size_t histogramsize = HISTOGRAM_LENGTH * sizeof(float);
  //@@ insert code here
  wbCheck(cudaMalloc((void **)&deviceInputImageData, imagesize * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutputImageData, imagesize * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceGreyImage, greysize * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceHistogram, histogramsize));
  wbCheck(cudaMalloc((void **)&deviceCDF, histogramsize));
  wbCheck(cudaMemcpy(deviceInputImageData, hostInputImageData, imagesize * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemset(deviceHistogram, 0, histogramsize));
  wbCheck(cudaMemset(deviceCDF, 0, histogramsize));
  size_t numblock = (greysize-1)/(BLOCK_SIZE) + 1;
  // Each data is in charge of one channel (namely three pixels)
  dim3 GreyGridDim(numblock, 1, 1);
  dim3 GreyBlockDim(BLOCK_SIZE, 1, 1);
  // Only one block with 256 elements is needed
  dim3 ScanGridDim(1, 1, 1);
  dim3 ScanBlockDim(SCAN_BLOCK_SIZE, 1, 1);
  greyscale<<<GreyGridDim, GreyBlockDim>>>(deviceInputImageData, deviceHistogram, greysize);
  scan<<<ScanGridDim, ScanBlockDim>>>(deviceHistogram, deviceCDF, greysize);
  recolor<<<GreyGridDim, GreyBlockDim>>>(deviceInputImageData, deviceOutputImageData, deviceCDF, imagesize);
  wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputImageData, imagesize * sizeof(float), cudaMemcpyDeviceToHost));
  wbCheck(cudaDeviceSynchronize());

  wbSolution(args, outputImage);

  //@@ insert code here
  wbCheck(cudaFree(deviceInputImageData));
  wbCheck(cudaFree(deviceOutputImageData));
  wbCheck(cudaFree(deviceGreyImage));
  wbCheck(cudaFree(deviceHistogram));
  wbCheck(cudaFree(deviceCDF));
  free(hostInputImageData);
  free(hostOutputImageData);


  return 0;
}
