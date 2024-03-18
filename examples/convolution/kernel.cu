#include "common.h"
#include "timer.h"

#define OUT_TILE_DIM 32 


__constant__ float mask_c[MASK_DIM][MASK_DIM];

__global__ void convolution_kernel()



// host function
void convolution_gpu(float mask[][MASK_DIM], float* input, float* output, unsigned int width, unsigned int height){


    // constant memory can only be copied from host to gpu
    // we can only allocate up to 64KB 
    cudaMemcpyToSymbol(mask_c, mask, MASK_DIM*MASK_DIM*sizeof(float));

}
