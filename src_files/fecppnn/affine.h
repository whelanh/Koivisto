//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_AFFINE_H
#define KOIVISTO_AFFINE_H

#include "Data.h"

#include <immintrin.h>
namespace nn {

void affine_transformation(Data* matrixData, Data* biasData, Data* inputData, Data* outputData);

#ifdef NN_TRAIN
void affine_transformation_backprop(Data* matrixData, Data* biasData, Data* inputData, Data* outputData, int threadID);
#endif

}    // namespace nn

#endif    // KOIVISTO_AFFINE_H
