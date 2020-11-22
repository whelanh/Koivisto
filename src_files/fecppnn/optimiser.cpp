//
// Created by Luecx on 21.11.2020.
//

#include "optimiser.h"

#include <iostream>

#ifdef NN_TRAIN

nn::Optimiser::Optimiser(double alpha) : alpha(alpha) {}
void nn::Optimiser::optimise(int size, nn::Data** weights, nn::Data** bias) {
    for(int i = 0; i < size; i++){
        for(int n = 0; n < weights[i]->size; n++){
            weights[i]->values[n] += -(weights[i]->getGradient(0)->get(n) * alpha);
        }
        for(int n = 0; n < bias[i]->size; n++){
            bias[i]->values[n] += -(bias[i]->getGradient(0)->get(n) * alpha);
        }
    }
}

#endif