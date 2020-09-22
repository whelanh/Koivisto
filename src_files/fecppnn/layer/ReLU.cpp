/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/


#include "ReLU.h"
void fecppnn::ReLU::compute() {
    
    static __m256 lower = _mm256_set1_ps(0);
    
    float* outputVals = getOutput()->getValues();
    float* inputVals  = getInput()->getValues();
    
    for(int i = 0; i < getOutput()->getSize(); i+=8){
        
        __m256 in  = _mm256_load_ps(&(inputVals[i]));
        
        __m256 out = _mm256_max_ps(in, lower);
        
        _mm256_store_ps(&(outputVals[i]), out);
    }
}
void fecppnn::ReLU::backprop() {
    
    float* outputGrads = getOutput()->getGradient()->getValues();
    float* outputVals  = getOutput()->getValues();
    float* inputGrads  = getInput()->getGradient()->getValues();
    
    for (int i = 0; i < getOutput()->getSize(); i++) {
        inputGrads[i] = outputVals[i] < 0 ? 0 : outputGrads[i];
    }
}

const std::string fecppnn::ReLU::name() { return "ReLU"; }
