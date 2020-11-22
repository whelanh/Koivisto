//
// Created by Luecx on 21.11.2020.
//

#include "activations.h"

#include <immintrin.h>

void nn::activate_relu(nn::Data* input, const nn::Data* output) {
    
    static __m256 lower = _mm256_set1_ps(0);

    float* outputVals = input->getValues();
    float* inputVals  = output->getValues();

    int size = input->getSize();
    if(size % 8 != 0){
        size -= size % 8;
    }

    for(int i = 0; i < size; i+=8){
        __m256 in  = _mm256_load_ps(&(inputVals[i]));
        __m256 out = _mm256_max_ps(in, lower);

        _mm256_store_ps(&(outputVals[i]), out);
    }
    for (int i = size; i < input->size; i++)
        output->values[i] = relu(input->values[i]);
}
void nn::activate_sigmoid(nn::Data* input, const nn::Data* output) {
    for (int i = 0; i < input->size; i++)
        output->values[i] = sigmoid(input->values[i]);
}
void nn::activate_null(nn::Data* input, const nn::Data* output) {
    for (int i = 0; i < input->size; i++)
        output->values[i] = input->values[i];
}
void nn::backprop_relu(nn::Data* activation, const nn::Data* output) {
#ifdef NN_TRAIN
    for (int i = 0; i < output->size; i++){
        output->getGradient(0)->get(i) = activation->getGradient(0)->get(i) * relu_prime(output->values[i]);
    }
#endif
}
void nn::backprop_sigmoid(nn::Data* activation, const nn::Data* output) {
#ifdef NN_TRAIN
    for (int i = 0; i < output->size; i++){
        output->getGradient(0)->get(i) = activation->getGradient(0)->get(i) * sigmoid_prime(output->values[i]);
    }
#endif
}
void nn::backprop_null(nn::Data* activation, const nn::Data* output) {
#ifdef NN_TRAIN
    for (int i = 0; i < output->size; i++){
        output->getGradient(0)->get(i) = activation->getGradient(0)->get(i);
    }
#endif
}
