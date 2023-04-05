#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#ifndef UTILS_H_
#define UTILS_H_
#include "../utils/utils.h"
#endif

__global__ void hist_kernel(int *hist, double *input, int input_size, double minVal, double maxVal, double range, int num_bins){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (i < input_size){
        double val = input[i];
        int bin = static_cast<int>((val - minVal)/range*num_bins);
        if (bin < 0) {
            bin = 0;
        }else if (bin >= num_bins){
            bin = num_bins -1;
        }
        
        atomicAdd(&hist[bin],1);
        i += stride;
    }
}

int main(int argc, char **argv){
    std::string filename = "../data/input.txt";
    int num_bins = 10;

    std::vector<double> *input = readInput<double>(filename);

    int input_size = input->size();
    double minVal = *std::min_element(input->data(), input->data()+input->size());
    double maxVal = *std::max_element(input->data(), input->data()+input->size());
    double range = maxVal - minVal;

    double *d_input;

    cudaMalloc(&d_input, input->size() * sizeof(double));

    cudaMemcpy(d_input, input->data(), input->size() * sizeof(double), cudaMemcpyHostToDevice);


    int *d_hist;
    cudaMalloc(&d_hist, num_bins * sizeof(int));

    cudaMemset(d_hist, 0, num_bins*sizeof(int));

    int blockSize = 1024;
    int numBlocks = (input->size() + blockSize - 1) / blockSize;
    
    hist_kernel<<<numBlocks, blockSize>>>(d_hist, d_input, input_size, minVal, maxVal, range, num_bins);


    int *h_hist = new int[num_bins];
    cudaMemcpy(h_hist, d_hist, num_bins*sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0 ; i < num_bins; i++){
        std::cout << "BIN[" << i << "]: " << h_hist[i] << std::endl;
    }

    cudaFree(d_input);
    cudaFree(d_hist);
    delete input;
    delete[] h_hist;
}
