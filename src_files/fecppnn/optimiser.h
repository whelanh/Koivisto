//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_OPTIMISER_H
#define KOIVISTO_OPTIMISER_H

#include "Data.h"
#ifdef NN_TRAIN

namespace nn{

struct Optimiser{
    
    double alpha;

    Optimiser(double alpha);

    void optimise(Data** weights, Data** bias);
};


struct Adam{
    
    private:
    double alpha = 0.001, beta1 = 0.9, beta2 =0.999, epsilon = 1e-8;
    int timeStep;
    Data* weights_fmv[LAYER_COUNT]{nullptr};
    Data* weights_smv[LAYER_COUNT]{nullptr};
    Data*    bias_fmv[LAYER_COUNT]{nullptr};
    Data*    bias_smv[LAYER_COUNT]{nullptr};
    
    void initVectors();
    
    void optimise(Data** weights, Data** bias);
    
    
};

}

#endif


#endif    // KOIVISTO_OPTIMISER_H
