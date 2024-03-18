#include "common.h"
#include "timer.h"

__global__ void blur_kernel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height){

    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    int outCol = blockIdx.x * blockDim.x + threadIdx.x; 

    if (outRow < height && outCol < width){

        unsigned int average = 0; 
        for (int inRow = outRow - BLUR_SIZE; inRow < outRow + BLUR_SIZE + 1; ++inRow ){
            for (int inCol = outCol - BLUR_SIZE; inCol < outCol + BLUR_SIZE + 1 ; ++inCol) {
                
                if (inRow >= 0 && inRow < height &&  inCol >= 0 && inCol < width){

                    average += image[inRow*width + inCol];
                }
            }
        }

        blurred[outRow*width + outCol] = (unsigned char)(average / ((BLUR_SIZE * 2 + 1) * (BLUR_SIZE * 2 + 1));

    }
    


}


void blur_gpu(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height){

    Timer timer; 

    // Allocate GPU memory
    startTime(&timer);
    unsigned char *image_d, *blurred_d;
    cudaMalloc((void**)&image_d, width*height*sizeof(unsigned char)); 
    cudaMalloc((void**)&blurred_d, width*height*sizeof(unsigned char));
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsed(timer, "Allocation time");

    // Copy data to GPU 
    startTime(&timer);
    cudaMemcpy(image_d, image, width*height*sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(time, "Copy to GPU time");

    // Call kernel 
    startTime(&timer);
    dim3 numThreadsPerBlock(16,16);
    dim3 numBlocks((widt + numThreadsPerBlock.x -1)/ numThreadsPerBlock.x, (height + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y);
    blur_kernel<<< numBlocks, numThreadsPerBlock >>> (image_d, blurred_d, width, height);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Kernel time", GREEN);

    // Copy data from GPU
    startTime(&timer);
    cudaMemcpy(blurred, blurred_d, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Copy from GPU time");

    startTime(&timer);
    cudaFree(image_d);
    cudaFree(blurred_d);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Deallocation time");

}
