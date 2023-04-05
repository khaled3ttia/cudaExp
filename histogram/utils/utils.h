#include <vector>
#include <fstream>
#include <iostream>

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


