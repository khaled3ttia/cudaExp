#include "common.h"
#include "timer.h"

#define TILE_DIM 32

__global__ void mm_kernel(float* A, float *B , float *C, unsigned int N){

    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    // iterate over tiles
    for (unsigned int tile = 0 ; tile < N/TILE_DIM; ++tile){
       // each thread load one element 
        A_s[threadIdx.y][threadIdx.x] = A[row*N + tile*TILE_DIM + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE_DIM+ threadIdx.y) *N+ col];

        __syncthreads();

        for (unsigned int i = 0 ; i < TILE_DIM; ++i){

            sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];

        }
        __syncthreads();
    }

    C[row*N + col] = sum;
}


void mm_gpu(float* A, float* B , float* C, unsigned int N){

    Timer timer;

    // Allocate GPU memory
    startTime(&timer);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, N*N*sizeof(float));
    cudaMalloc((void**)&B_d, N*N*sizeof(float));
    cudaMalloc((void**)&C_d, N*N*sizeof(float));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Allocation time");

    // Copy data to GPU
    startTime(&timer);
    cudaMemcpy(A_d, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy to GPU time");


    // Call kernel
    startTime(&timer);
    dim3 numThreadsPerBlock(32,32);
    dim3 numBlocks((N + numThreadsPerBlock.x - 1)/numThreadsPerBlock.x , (N+ numThreadsPerBlock.y - 1)/numThreadsPerBlock.y);
    mm_kernel<<< numBlocks, numThreadsPerBlock >>> (A_d, B_d, C_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(C, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    startTime(&timer);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");
}
