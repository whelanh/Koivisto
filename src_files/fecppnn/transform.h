//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_TRANSFORM_H
#define KOIVISTO_TRANSFORM_H

#include "../Board.h"
#include "Data.h"
#include "config.h"
#include "sample.h"

namespace nn{

void transform(Sample* in, Data* weights, Data* bias, Data* output);
void transform_backprop(Sample* in, Data* weights, Data* bias, Data* output, int threadID);

}

#endif    // KOIVISTO_TRANSFORM_H
