/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#include "Sigmoid.h"
void fecppnn::Sigmoid::backprop() {
    
    float* outputGrads = getOutput()->getGradient()->getValues();
    float* outputVals  = getOutput()->getValues();
    float* inputGrads  = getInput()->getGradient()->getValues();
    
    for (int i = 0; i < getOutput()->getSize(); i++) {
        inputGrads[i] = outputGrads[i] * (outputVals[i] * (1 - outputVals[i]));
    }
}
void fecppnn::Sigmoid::compute() {
    float* outputVals = getOutput()->getValues();
    float* inputVals  = getInput()->getValues();
    
    for (int i = 0; i < getOutput()->getSize(); i++) {
        outputVals[i] = 1.0 / (1 + exp(-inputVals[i]));
    }
}
void fecppnn::Sigmoid::collectOptimisableData(std::vector<Data*>& vec) {}

const std::string fecppnn::Sigmoid::name() { return "Sigmoid"; }
