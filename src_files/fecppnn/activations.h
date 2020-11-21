//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_ACTIVATIONS_H
#define KOIVISTO_ACTIVATIONS_H

#include <math.h>
#include "Data.h"
namespace nn{


inline float relu(float x) {
    return fmaxf(0.0, x);
}

inline float relu_prime(float x) {
    return x > 0.0 ? 1.0 : 0.0;
}

inline float sigmoid(float x) {
    return 1.0 / (1.0 + expf(x));
}

inline float sigmoid_prime(float x) {
    float sigm = sigmoid(x);
    return sigm * (1.0 - sigm);
}


void activate_relu(nn::Data *input, const nn::Data *output);

void activate_sigmoid(nn::Data *input, const nn::Data *output);

void activate_null(nn::Data *input, const nn::Data *output);


void backprop_relu(nn::Data *activation, const nn::Data *output);

void backprop_sigmoid(nn::Data *activation, const nn::Data *output);

void backprop_null(nn::Data *activation, const nn::Data *output);

}

#endif    // KOIVISTO_ACTIVATIONS_H
