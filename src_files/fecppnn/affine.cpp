//
// Created by Luecx on 21.11.2020.
//
#include "affine.h"

#include <iostream>
void nn::affine_transformation(nn::Data* matrixData, nn::Data* biasData, nn::Data* inputData, nn::Data* outputData) {
    
    int inputSize = inputData->getSize();
    
    float* outputValues = outputData->values;
    float* inputValues  = inputData->values;
    float* bias         = biasData->values;
    float* weights      = matrixData->values;
    
    int size = outputData->size;
    if(size % 8 != 0){
        size -= size % 8;
    }
    for (int row = 0; row < size; row += 8) {
        
        __m256 biasV = _mm256_load_ps(&(bias[row]));
        
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        for (int col = 0; col < inputSize; col += 8) {
            __m256 vec = _mm256_load_ps(&inputValues[col]);
            
            __m256 mat0 = _mm256_load_ps(&weights[col + inputSize * row]);
            __m256 mat1 = _mm256_load_ps(&weights[col + inputSize * (row + 1)]);
            __m256 mat2 = _mm256_load_ps(&weights[col + inputSize * (row + 2)]);
            __m256 mat3 = _mm256_load_ps(&weights[col + inputSize * (row + 3)]);
            __m256 mat4 = _mm256_load_ps(&weights[col + inputSize * (row + 4)]);
            __m256 mat5 = _mm256_load_ps(&weights[col + inputSize * (row + 5)]);
            __m256 mat6 = _mm256_load_ps(&weights[col + inputSize * (row + 6)]);
            __m256 mat7 = _mm256_load_ps(&weights[col + inputSize * (row + 7)]);
            
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
        
        _mm256_store_ps(&outputValues[row], acc0);
    }
    
    
    for (int row = size; row < outputData->size; row++){
        __m256 acc0 = _mm256_setzero_ps();
        for (int col = 0; col < inputSize; col += 8) {
            __m256 vec = _mm256_load_ps(&inputValues[col]);
            __m256 mat0 = _mm256_load_ps(&weights[col + inputSize * row]);
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(mat0, vec));
        }
        outputData->values[row] = acc0[0] + acc0[1] + acc0[2] + acc0[3] +
                                  acc0[4] + acc0[5] + acc0[6] + acc0[7] +
                                  bias[row];
    }
}

#ifdef NN_TRAIN
void nn::affine_transformation_backprop(nn::Data* matrixData, nn::Data* biasData, nn::Data* inputData,
                                        nn::Data* outputData, int threadID) {
    
    for(int o = 0; o < outputData->size; o++){
        biasData->getGradient(threadID)->get(o) += outputData->getGradient(0)->get(o);
    }
    
    for(int i = 0; i < inputData->size; i++){
        double sum = 0;
        for(int o = 0; o < outputData->size; o++){
            matrixData->getGradient(threadID)->get(i,o) += outputData->getGradient(0)->get(o) * inputData->get(i);
            sum += outputData->getGradient(0)->get(o) * matrixData->get(i, o);
        }
        
        inputData->getGradient(0)->get(i) = sum;
        
    }
    
}
#endif
