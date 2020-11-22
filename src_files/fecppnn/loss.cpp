//
// Created by Luecx on 21.11.2020.
//
#include "loss.h"

#include <iostream>

#ifdef NN_TRAIN
float loss_l2(nn::Data* output, const nn::Data* target) {
    
    float loss = 0;
    for(int i = 0; i < output->size; i++){
        float difference = output->get(i) - target->get(i);
        loss += difference * difference;
        output->getGradient(0)->get(i) = difference * 2;
    }
    return loss;
    
}

float loss_l1(nn::Data* output, const nn::Data* target) {
    
    float loss = 0;
    for(int i = 0; i < output->size; i++){
        float difference = output->get(i) - target->get(i);
        loss += abs(difference);
        output->getGradient(0)->get(i) = difference > 0 ? 1:-1;
    }
    return loss;
    
}
#endif