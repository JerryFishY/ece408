// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float* aux, float *output, int len, bool NeedAux) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  int tx = threadIdx.x;
  int gx = blockIdx.x*blockDim.x*2 + tx;
  __shared__ float SharedInput[2*BLOCK_SIZE];
  // load data into shared memory
  if(gx < len) {
    SharedInput[tx] = input[gx];
  } else {
    SharedInput[tx] = 0;
  }
  if(gx + blockDim.x < len) {
    SharedInput[tx + blockDim.x] = input[gx + blockDim.x];
  } else {
    SharedInput[tx + blockDim.x] = 0;
  }
  __syncthreads();

  // reduction
  for (size_t stride = 1;stride<=BLOCK_SIZE;stride*=2) {
    int index = (tx+1)*stride*2-1;
    if(index < 2*BLOCK_SIZE) {
      SharedInput[index] = SharedInput[index-stride]+SharedInput[index];
    }
    __syncthreads();
  }
  // distribution
  for (size_t stride = BLOCK_SIZE / 2;stride > 0; stride /= 2) {
    int index = (tx+1)*stride*2-1;
    if(index+stride < 2*BLOCK_SIZE) {
      SharedInput[index+stride] += SharedInput[index];
    }
    __syncthreads();
  }
    // Wirte back to global
    if (gx < len)
    {
      output[gx] = SharedInput[tx];
    }
  if(gx + blockDim.x < len) {
    output[gx + blockDim.x] = SharedInput[tx + blockDim.x];
  }
  // Wirte to aux
  if(tx == 0 && NeedAux) {
    aux[blockIdx.x] = SharedInput[2*BLOCK_SIZE - 1];
  }
}
// Note each thread takes care of two elements
__global__ void add(float *input, float *aux, int len) {
  if(blockIdx.x == 0) {
    return;
  }
  int tx = threadIdx.x;
  int gx = blockIdx.x*blockDim.x + tx;
  if(gx < len) {
    input[gx] += aux[blockIdx.x-1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceaux;
  float *devicescanaux;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);
  // Note: the last element of the last block donot need to be scanned, the first block donot need to be postprocessed
  size_t numblock = (numElements-1)/(BLOCK_SIZE<<1) + 1; // One block take care of two elements
  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceaux, (numblock)*sizeof(float)));
  wbCheck(cudaMalloc((void **)&devicescanaux, (numblock)*sizeof(float)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(deviceaux, 0, (numblock) * sizeof(float)));
  wbCheck(cudaMemset(devicescanaux, 0, (numblock) * sizeof(float)));

  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  // first and third stage dim
  dim3 ScanGridDim(numblock, 1, 1);
  dim3 ScanBlockDim(BLOCK_SIZE, 1, 1);
  // second stage dim
  dim3 Scan2GridDim(1, 1, 1);
  dim3 Scan2BlockDim(BLOCK_SIZE, 1, 1);
  // third stage dim
  dim3 AddGridDim(numblock, 1, 1);
  dim3 AddBlockDim(BLOCK_SIZE<<1, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<ScanGridDim, ScanBlockDim>>>(deviceInput,deviceaux, deviceOutput, numElements,true);
  scan<<<Scan2GridDim, Scan2BlockDim>>>(deviceaux, nullptr, devicescanaux, numblock,false);
  add<<<AddGridDim, AddBlockDim>>>(deviceOutput, devicescanaux, numElements);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
