/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#include "Concat.h"

fecppnn::Concat::Concat(Layer* prev1, Layer* prev2)
    : Layer(prev1->getOutput()->getSize() + prev2->getOutput()->getSize()) {
    connect(prev1);
    connect(prev2);
}

void fecppnn::Concat::compute() {

    float* in1 = getInput(flipInputs)->getValues();
    float* in2 = getInput(1 - flipInputs)->getValues();

    float* out = getOutput()->getValues();

    int in1Size = getInput(flipInputs)->getSize();
    int in2Size = getInput(1 - flipInputs)->getSize();

    memcpy(&out[0], in1, sizeof(float) * in1Size);
    memcpy(&out[in1Size], in2, sizeof(float) * in2Size);
}
void fecppnn::Concat::backprop() {
    float* in1 = getInput(flipInputs)->getGradient()->getValues();
    float* in2 = getInput(1 - flipInputs)->getGradient()->getValues();

    float* out = getOutput()->getGradient()->getValues();

    int in1Size = getInput(flipInputs)->getSize();
    int in2Size = getInput(1 - flipInputs)->getSize();

    memcpy(in1, &out[0], sizeof(float) * in1Size);
    memcpy(in2, &out[in1Size], sizeof(float) * in2Size);
}

const std::string fecppnn::Concat::name() { return "Concat"; }
bool              fecppnn::Concat::isFlipInputs() const { return flipInputs; }
void              fecppnn::Concat::setFlipInputs(bool flipInputs) { Concat::flipInputs = flipInputs; }
