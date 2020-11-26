//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_STRUCTURE_H
#define KOIVISTO_STRUCTURE_H

#include "config.h"
#include "Layer.h"
#include "activations.h"
#include "loss.h"
#include "Optimiser.h"

#define LAYER_COUNT   3

#define RELATIVE 1
#define FULL 2

#define NN_TYPE FULL

namespace nn{

static std::string weightsFile = "nn4.bin";
static bool        loadWeights = true;

static Layer layers[LAYER_COUNT]{
    {64*12,128, &activate_relu, &backprop_relu},
    {128  , 32, &activate_relu, &backprop_relu},
    {32   ,  1, &activate_null, &backprop_null},
};

#ifdef NN_TRAIN
static Loss lossFunction = &loss_l2;

static Adam optimiser{0};
#endif



}

#endif    // KOIVISTO_STRUCTURE_H
