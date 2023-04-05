#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <string>
#include <algorithm>

template<typename T>
std::vector<T>* readInput(const std::string& filename){
    std::ifstream file(filename);
    if (!file.is_open()){
        std::cerr << "Error opening file " << filename << std::endl;
        return nullptr;
    }

    std::vector<T> *input = new std::vector<T>();
    T num; 

    while (file >> num){
        input->push_back(num);
    }

    file.close();
    return input;
}

template<typename T>
int* computeHist(T* input, size_t input_size, unsigned int num_bins){

    int *hist = new int[num_bins];
    std::fill(hist, hist+num_bins, 0);
    T minVal = *std::min_element(input, input+input_size);
    T maxVal = *std::max_element(input, input+input_size);
    T range = maxVal - minVal;

    int i = 0; 
    while (i < input_size){
        T val = input[i];
        int bin = static_cast<int>((val - minVal)/ range*num_bins);
        if (bin < 0){
            bin = 0;
        }else if (bin >= num_bins){
            bin = num_bins - 1;
        }

        hist[bin]++;
        i++;
    }

    return hist;
}

void help(char **argv) {

    std::cout << "Usage: " << std::endl;
    std::cout << argv[0] << " -i input.txt -b 10" << std::endl;
    std::cout << "input.txt is the path to the input file. Each line in that file should contain only one number" << std::endl;;
    std::cout << "10 is the number of bins in the histogram. Must be an integer." << std::endl;
    exit(0);
}
int main(int argc, char **argv){

    std::string filename = "../data/input.txt";
    int num_bins = 10;

    int opt;
    while ((opt = getopt(argc, argv, "hi:b:")) != -1){
        switch (opt){
            case 'i':
                filename = optarg;
                break;
            case 'b': 
                num_bins = atoi(optarg);
                break;
            case 'h':
                help(argv);
        }
    }

    std::vector<double> *input = readInput<double>(filename);
    if (input == nullptr){
        return 1;
    } 
    int *hist = computeHist(input->data(), input->size(), num_bins);

    int sum = 0;
    for (int i = 0 ; i < num_bins; i++){
        std::cout << "BIN[" << i << "]: " << hist[i] << std::endl;
        sum += hist[i];
    }

    if (sum == input->size()){
        std::cout << "Verified successfully!" << std::endl;
    }else{
        std::cout << "Verification failed :(" << std::endl;
    }

    delete[] hist;
    delete input;

}
