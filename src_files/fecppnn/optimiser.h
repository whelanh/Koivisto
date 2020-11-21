//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_OPTIMISER_H
#define KOIVISTO_OPTIMISER_H

#include "Network.h"
#include "config.h"

#ifdef NN_TRAIN

namespace nn{

struct Optimiser{
    
    Network* net;
    double alpha;

    Optimiser(double alpha);

    void optimise();
};

}

#endif


#endif    // KOIVISTO_OPTIMISER_H
