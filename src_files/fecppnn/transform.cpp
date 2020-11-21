//
// Created by Luecx on 21.11.2020.
//
#include "transform.h"
#if NN_TYPE == FULL
void nn::transform(Sample* in, nn::Data* weights, Data* bias, nn::Data* output) {
    
    for(int i = 0; i < output->size; i++){
        (*output)(i) = (*bias)(i);
    }
    
    for(uint16_t &index:in->indices){
        for(int n = 0; n < output->size; n++){
            (*output)(n) += (*weights).values[index * output->size + n];
        }
    }
}
#ifdef NN_TRAIN
void nn::transform_backprop(Sample* in, nn::Data* weights, Data* bias, nn::Data* output, int threadID) {
    for(int i = 0; i < output->size; i++){
        bias->getGradient(threadID)->get(i) += output->getGradient(0)->get(i);
    }
    
    for(uint16_t &index:in->indices){
        for(int n = 0; n < output->size; n++){
            weights->getGradient(threadID)->get(index * output->size + n) += output->getGradient(0)->get(n);
        }
    }
}
#endif
#else

#endif
