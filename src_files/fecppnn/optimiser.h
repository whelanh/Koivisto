//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_OPTIMISER_H
#define KOIVISTO_OPTIMISER_H

#include "config.h"
#include "Data.h"

#ifdef NN_TRAIN

namespace nn{

struct Optimiser{
    
    double alpha;

    Optimiser(double alpha);

    void optimise(int size, Data** weights, Data** bias);
};

}

#endif


#endif    // KOIVISTO_OPTIMISER_H
