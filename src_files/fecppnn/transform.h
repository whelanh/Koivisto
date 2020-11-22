//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_TRANSFORM_H
#define KOIVISTO_TRANSFORM_H

#include "../Board.h"
#include "Data.h"
#include "config.h"
#include "sample.h"
#include "immintrin.h"

namespace nn{

void affine_transformation_input(Sample* in, Data* weights, Data* bias, Data* output);
void affine_transformation_input_backprop(Sample* in, Data* weights, Data* bias, Data* output, int threadID);

}

#endif    // KOIVISTO_TRANSFORM_H
