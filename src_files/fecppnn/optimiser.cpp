//
// Created by Luecx on 21.11.2020.
//

#include "optimiser.h"

#ifdef NN_TRAIN

nn::Optimiser::Optimiser(double alpha) : alpha(alpha) {}

void nn::Optimiser::optimise() {

    for(int i = 0; i < net->size; i++){
        for(int n = 0; n < net->weights[i]->size; n++){
            net->weights[i]->values[n] += -(net->weights[i]->getGradient(0)->get(n) * alpha);
        }
        for(int n = 0; n < net->biases[i]->size; n++){
            net->biases[i]->values[n] += -(net->biases[i]->getGradient(0)->get(n) * alpha);
        }
    }

}
#endif