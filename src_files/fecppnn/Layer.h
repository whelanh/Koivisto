//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_LAYER_H
#define KOIVISTO_LAYER_H

#include "activations.h"
#include "transform.h"

namespace nn {

struct Layer {

    int inputSize;
    int outputSize;
    
    Activation activation;
    BackProp   backprop;

    Layer(int inputSize, int outputSize, void (*activation)(Data*, const Data*), void (*backprop)(Data*, const Data*))
        : inputSize(inputSize),
          outputSize(outputSize),
          activation(activation),
          backprop(backprop){}
};
}    // namespace nn

#endif    // KOIVISTO_LAYER_H
