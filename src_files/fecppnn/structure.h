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



namespace nn{

static std::string weightsFile = "nn4.bin";
static bool        loadWeights = true;



#ifdef NN_TRAIN
static Loss lossFunction = &loss_l2;

static Optimiser optimiser{0.001/NN_BATCH_SIZE};
#endif



}

#endif    // KOIVISTO_STRUCTURE_H
