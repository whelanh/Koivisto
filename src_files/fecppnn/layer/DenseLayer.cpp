
/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/


#include "DenseLayer.h"


fecppnn::DenseLayer::DenseLayer(Layer* previousLayer, int size) : Layer(size) {
    weights = new Data(previousLayer->getOutput()->getSize(), size, true);
    bias    = new Data(size, true);
    
    float wBound = 1 / sqrt(previousLayer->getOutput()->getSize());
    
    weights->randomise(-wBound, +wBound);
    bias->randomise(-wBound, +wBound);
    
    this->connect(previousLayer);
}

void fecppnn::DenseLayer::compute() {
    
    Data* l_output = getOutput();
    Data* l_input  = getInput();
    
    int inputSize = l_input->getSize();
    
    float* l_outputValues = l_output->getValues();
    float* l_inputValues  = l_input->getValues();
    float* l_bias         = this->bias->getValues();
    float* l_weights      = this->weights->getValues();
    
    for (int row = 0; row < l_output->getSize(); row += 8) {
        
        __m256 biasV = _mm256_load_ps(&(l_bias[row]));
        
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        for (int col = 0; col < inputSize; col += 8) {
            __m256 vec = _mm256_load_ps(&l_inputValues[col]);
            
            __m256 mat0 = _mm256_load_ps(&l_weights[col + inputSize * row]);
            __m256 mat1 = _mm256_load_ps(&l_weights[col + inputSize * (row + 1)]);
            __m256 mat2 = _mm256_load_ps(&l_weights[col + inputSize * (row + 2)]);
            __m256 mat3 = _mm256_load_ps(&l_weights[col + inputSize * (row + 3)]);
            __m256 mat4 = _mm256_load_ps(&l_weights[col + inputSize * (row + 4)]);
            __m256 mat5 = _mm256_load_ps(&l_weights[col + inputSize * (row + 5)]);
            __m256 mat6 = _mm256_load_ps(&l_weights[col + inputSize * (row + 6)]);
            __m256 mat7 = _mm256_load_ps(&l_weights[col + inputSize * (row + 7)]);
            
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(mat0, vec));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(mat1, vec));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(mat2, vec));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(mat3, vec));
            acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(mat4, vec));
            acc5 = _mm256_add_ps(acc5, _mm256_mul_ps(mat5, vec));
            acc6 = _mm256_add_ps(acc6, _mm256_mul_ps(mat6, vec));
            acc7 = _mm256_add_ps(acc7, _mm256_mul_ps(mat7, vec));
        }
        
        acc0 = _mm256_hadd_ps(acc0, acc1);
        acc2 = _mm256_hadd_ps(acc2, acc3);
        
        acc4 = _mm256_hadd_ps(acc4, acc5);
        acc6 = _mm256_hadd_ps(acc6, acc7);
        
        acc0 = _mm256_hadd_ps(acc0, acc2);
        acc4 = _mm256_hadd_ps(acc4, acc6);
        
        __m128 sumabcd1 = _mm256_extractf128_ps(acc0, 0);
        __m128 sumabcd2 = _mm256_extractf128_ps(acc0, 1);
        __m128 sumefgh1 = _mm256_extractf128_ps(acc4, 0);
        __m128 sumefgh2 = _mm256_extractf128_ps(acc4, 1);
        
        sumabcd1 = _mm_add_ps(sumabcd1, sumabcd2);
        sumefgh1 = _mm_add_ps(sumefgh1, sumefgh2);
        
        acc0 = _mm256_insertf128_ps(_mm256_castps128_ps256(sumabcd1), sumefgh1, 1);
        
        acc0 = _mm256_add_ps(biasV, acc0);
        
        _mm256_store_ps(&l_outputValues[row], acc0);
    }
}
void fecppnn::DenseLayer::backprop() {
    Data* l_output = getOutput();
    Data* l_input  = getInput();
    
    int inputSize = l_input->getSize();
    
    if(l_input->hasGradient()){
        float* l_outputGrads  = l_output->getGradient()->getValues();
        float* l_inputGrads   = getInput()->getGradient()->getValues();
        float* l_inputValues  = getInput()->getValues();
        float* l_weights      = this->weights->getValues();
        float* l_weightsGrads = this->weights->getGradient()->getValues();
        float* l_biasGrads    = this->bias->getGradient()->getValues();
        
        
        //TODO optimise this similar to the forward pass
        for (int o = 0; o < l_output->getSize(); o++) {
            
            __m256 outputGrad = _mm256_set1_ps(l_outputGrads[o]);
            l_biasGrads[o] += l_outputGrads[o];
            
            for (int i = 0; i < inputSize; i += 8) {
                __m256 mat = _mm256_load_ps(&l_weights[i + inputSize * o]);
                _mm256_store_ps(&l_inputGrads[i], _mm256_mul_ps(mat, outputGrad));
                
                __m256 weightGrads = _mm256_load_ps(&l_weightsGrads[i + inputSize * o]);
                __m256 inputs      = _mm256_load_ps(&l_inputValues[i]);
                
                weightGrads = _mm256_add_ps(weightGrads, _mm256_mul_ps(outputGrad, inputs));
                _mm256_store_ps(&l_weightsGrads[i + inputSize * o], weightGrads);
            }
        }
    }else{
        float* l_outputGrads  = l_output->getGradient()->getValues();
        float* l_inputValues  = getInput()->getValues();
        float* l_weightsGrads = this->weights->getGradient()->getValues();
        float* l_biasGrads    = this->bias->getGradient()->getValues();
        
        for (int o = 0; o < l_output->getSize(); o++) {
            
            __m256 outputGrad = _mm256_set1_ps(l_outputGrads[o]);
            l_biasGrads[o] += l_outputGrads[o];
            
            for (int i = 0; i < inputSize; i += 8) {
                __m256 weightGrads = _mm256_load_ps(&l_weightsGrads[i + inputSize * o]);
                __m256 inputs      = _mm256_load_ps(&l_inputValues[i]);
                
                weightGrads = _mm256_add_ps(weightGrads, _mm256_mul_ps(outputGrad, inputs));
                _mm256_store_ps(&l_weightsGrads[i + inputSize * o], weightGrads);
            }
        }
    }
    
    
}
fecppnn::Data* fecppnn::DenseLayer::getWeights() const { return weights; }
void  fecppnn::DenseLayer::setWeights(Data* p_weights) { DenseLayer::weights = p_weights; }
fecppnn::Data* fecppnn::DenseLayer::getBias() const { return bias; }
void  fecppnn::DenseLayer::setBias(Data* p_bias) { DenseLayer::bias = p_bias; }
const std::string fecppnn::DenseLayer::name() { return "DenseLayer"; }
