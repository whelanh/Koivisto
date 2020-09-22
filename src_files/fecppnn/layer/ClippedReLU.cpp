/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#include "ClippedReLU.h"

fecppnn::ClippedReLU::ClippedReLU(Layer* prevLayer) : Layer(prevLayer->getOutput()->getSize()) { this->connect(prevLayer); }


void fecppnn::ClippedReLU::compute() {
    
    static __m256 lower = _mm256_set1_ps(0);
    static __m256 upper = _mm256_set1_ps(1);
    
    
    float* outputVals = getOutput()->getValues();
    float* inputVals  = getInput()->getValues();
    
    for(int i = 0; i < getOutput()->getSize(); i+=8){
        
        __m256 out = _mm256_load_ps(&(outputVals[i]));
        __m256 in  = _mm256_load_ps(&(inputVals[i]));
        
        out = _mm256_max_ps(in, lower);
        out = _mm256_min_ps(out, upper);
        
        _mm256_store_ps(&(outputVals[i]), out);
    }
}
void fecppnn::ClippedReLU::backprop() {
    
    float* outputGrads = getOutput()->getGradient()->getValues();
    float* outputVals  = getOutput()->getValues();
    float* inputGrads  = getInput()->getGradient()->getValues();
    
    for (int i = 0; i < getOutput()->getSize(); i++) {
        inputGrads[i] = outputVals[i] > 0 && outputVals[i] < 1 ? outputGrads[i] : 0;
    }
}

const std::string fecppnn::ClippedReLU::name() { return "ClippedReLU"; }
