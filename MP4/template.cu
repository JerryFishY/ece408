// Input parallel is used for this MP.
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
constexpr int PADED_TILE_SIZE = 10; // 1+8+1 with padding around the block
constexpr int TILE_SIZE = 8; // 8 is the tile size
//@@ Define constant memory for device kernel here
__constant__ float constant_kernel[27];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  __shared__ float N_ds[PADED_TILE_SIZE][PADED_TILE_SIZE][PADED_TILE_SIZE];
  float Pvalue = 0.0;
  // Perform loading
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int gx = blockIdx.x * TILE_SIZE + tx - 1;
  int gy = blockIdx.y * TILE_SIZE + ty - 1;
  int gz = blockIdx.z * TILE_SIZE + tz - 1;
  // Load the data
  if(gx<0 || gy<0 || gz<0 || gx>=x_size || gy>=y_size || gz>=z_size){
    N_ds[tz][ty][tx] = 0.0;
  }else{
    N_ds[tz][ty][tx] = input[gz * y_size * x_size + gy * x_size + gx];
  }
  __syncthreads();

  //Computational part
  if(tx<TILE_SIZE && ty<TILE_SIZE && tz<TILE_SIZE && gx+1<x_size && gy+1<y_size && gz+1<z_size){
    // Perform the convolution
    #pragma unroll
    for (int i = 0;i<3;i++){
      for (int j = 0;j<3;j++){
        for (int k = 0;k<3;k++){
          Pvalue += N_ds[tz+i][ty+j][tx+k] * constant_kernel[i*3*3+j*3+k];
        }
      }
    }
    output[(gz+1) * y_size * x_size + (gy+1) * x_size + gx+1] = Pvalue;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first three elements were the dimensions
  cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput,hostInput+3,(inputLength - 3) * sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(constant_kernel,hostKernel,27*sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimGrid(ceil(x_size/8.0),ceil(y_size/8.0),ceil(z_size/8.0));
  dim3 DimBlock(PADED_TILE_SIZE,PADED_TILE_SIZE,PADED_TILE_SIZE);
  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid,DimBlock>>>(deviceInput,deviceOutput,z_size,y_size,x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  cudaMemcpy(hostOutput+3,deviceOutput,(inputLength-3) * sizeof(float),cudaMemcpyDeviceToHost);

  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;

  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
