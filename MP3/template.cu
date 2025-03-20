
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
// The dimension is (M,K) @ (K,N)
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  //@@ first assume all blocks are square blocks
  constexpr size_t Tile_M = 16;
  constexpr size_t Tile_N = 16;
  constexpr size_t Tile_K = 16;
  size_t Tile_K_num = (numAColumns-1)/Tile_K+1;
  
  int gx = blockDim.x*blockIdx.x + threadIdx.x;
  int gy = blockDim.y*blockIdx.y + threadIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  float Cvalue = 0.0;

  __shared__ float subTileA[Tile_M][Tile_K];
  __shared__ float subTileB[Tile_K][Tile_N];

  // Iterate over all tiles
  for (int k = 0; k < Tile_K_num;k++){
    // Load the subtiles
    if(gy<numARows && k*Tile_K+tx<numAColumns){
      subTileA[ty][tx] = A[gy*numAColumns + k*Tile_K + tx];
    }else{
      subTileA[ty][tx] = 0.0;
    }
    if(gx<numBColumns && k*Tile_K+ty<numBRows){
      subTileB[ty][tx] = B[(k*Tile_K+ty)*numBColumns + gx];
    }else{
      subTileB[ty][tx] = 0.0;
    }
    __syncthreads();
    // Compute the tile
    for(int i = 0; i<Tile_K;i++){
      Cvalue += subTileA[ty][i]*subTileB[i][tx];
    }
    __syncthreads();
  } 
  if(gy<numCRows && gx<numCColumns){
    C[gy*numCColumns + gx] = Cvalue;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = 0;
  numCColumns = 0;
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  wbTime_stop(Generic, "Importing data and creating memory on host");
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float));
  cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float));
  cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  size_t sizeA = numARows * numAColumns * sizeof(float);
  size_t sizeB = numBRows * numBColumns * sizeof(float);
  size_t sizeC = numCRows * numCColumns * sizeof(float);

  cudaMemcpy(deviceA,hostA,sizeA,cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB,hostB,sizeB,cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 GridDim((numCColumns-1)/16+1, (numCRows-1)/16+1, 1);
  dim3 BlockDim(16, 16, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<GridDim,BlockDim>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC,deviceC,sizeC,cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
