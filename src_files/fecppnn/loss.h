//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_LOSS_H
#define KOIVISTO_LOSS_H

#include "Data.h"
#include "config.h"

#ifdef NN_TRAIN
float loss_l2(nn::Data *output, const nn::Data *target);
float loss_l1(nn::Data *output, const nn::Data *target);
#endif
#endif    // KOIVISTO_LOSS_H
