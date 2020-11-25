//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_CONFIG_H
#define KOIVISTO_CONFIG_H

#include "Layer.h"
namespace nn{
    class   Data;
    struct  Optimiser;
}

typedef void  (*Activation) (nn::Data* output, const nn::Data* activation);
typedef void  (*BackProp)   (nn::Data* output, const nn::Data* activation);
typedef float (*Loss)       (nn::Data* output, const nn::Data* target);


#define NN_TRAIN
#ifndef NN_TRAIN
#define NN_RUN
#endif

#define NN_THREADS 16
#define NN_BATCH_SIZE 512

#define RELATIVE 1
#define FULL 2

#define NN_TYPE FULL
#define LAYER_COUNT   3

static nn::Layer layers[LAYER_COUNT]{
    {64*12,128, &activate_relu, &backprop_relu},
    {128  , 32, &activate_relu, &backprop_relu},
    {32   ,  1, &activate_null, &backprop_null},
};




#endif    // KOIVISTO_CONFIG_H
