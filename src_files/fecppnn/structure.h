//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_STRUCTURE_H
#define KOIVISTO_STRUCTURE_H

#include "Layer.h"
#include "activations.h"
#include "loss.h"

#define LAYER_COUNT 3

namespace nn{

static Layer layers[LAYER_COUNT]{
    {12*64  , 32, &activate_relu, &backprop_relu},
    {32     , 32, &activate_relu, &backprop_relu},
    {32     ,  1, &activate_null, &backprop_null},
};

#ifdef NN_TRAIN
static Loss lossFunction = &loss_l2;
#endif

}

#endif    // KOIVISTO_STRUCTURE_H
