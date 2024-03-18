#include "timer.h"

__global__ void vecadd_kernel(float* x, float* y, float* z, int N){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) z[i] = x[i] + y[i];
    
}

void vecadd_cpu(float* x,float* y, float* z, int N){
    for (unsigned int = 0; i < N; ++i){
        z[i] = x[i] + y[i];
    }
}

void vecadd_gpu(float* x, float* y, float* z, int N){

    // Allocate GPU memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));
    

    // Copy to the GPU
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Call a GPU kernel function (launch a grid of threads)
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    Timer timer;
    startTime(&timer);
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
    cudaDeviceSynchronize();
    stopTime(&timer);

    printElapsedTime(time, "GPU kernel time", GREEN);

    // Copy from the GPU
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);


    // Deallocate GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main(int argc, char** argv){

    cudaDeviceSynchronize();

    Timer timer;
    unsigned int N = (argc > 1)?(atoi(argv[1])):(1<<25);

    float *x = (float*) malloc(N*sizeof(float));
    float *y = (float*) malloc(N*sizeof(float));
    float *z = (float*) malloc(N*sizeof(float));

    for (unsigned int i = 0 ; i < N; ++i) {
        x[i] = rand();
        y[i] = rand();

    }

    startTime(&timer);
    vecadd_cpu(x,y,z,N);
    stopTimer(&timer);

    printElapsedTime(timer, "CPU time", CYAN);

    startTime(&timer);
    vecadd_gpu(x,y,z,N);
    stopTime(&timer);

    printElapsedTime(time, "GPU time", DGREEN);

    free(x);
    free(y);
    free(z);

    return 0;
}
