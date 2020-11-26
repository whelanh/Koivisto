//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_OPTIMISER_H
#define KOIVISTO_OPTIMISER_H

#include "Data.h"

#include <vector>
#include <cmath>
#ifdef NN_TRAIN

namespace nn{

struct Optimiser{
    
    double alpha;

    Optimiser(double alpha);

    void optimise(int size, Data** weights, Data** bias);
};


struct Adam{
    
    private:
    double alpha = 0.001, beta1 = 0.9, beta2 =0.999, epsilon = 1e-8;
    int timeStep;
    
    
    std::vector<Data> weights_fmv;
    std::vector<Data> weights_smv;
    std::vector<Data>    bias_fmv;
    std::vector<Data>    bias_smv;
    
    void initVectors(int size, Data** weights, Data** bias);
    
    public:
    Adam(double alpha=0.001, double beta1=0.9, double beta2=0.999, double epsilon=1e-8);

    void optimise(int size, Data** weights, Data** bias);
    
    
};

}

#endif


#endif    // KOIVISTO_OPTIMISER_H
