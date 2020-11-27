//
// Created by Luecx on 21.11.2020.
//

#include "optimiser.h"

#include <iostream>

#ifdef NN_TRAIN

nn::Optimiser::Optimiser(double alpha) : alpha(alpha) {}
void nn::Optimiser::optimise(int size, nn::Data** weights, nn::Data** bias) {
    for(int i = 0; i < size; i++){
        for(int n = 0; n < weights[i]->size; n++){
            weights[i]->values[n] += -(weights[i]->getGradient(0)->get(n) * alpha);
        }
        for(int n = 0; n < bias[i]->size; n++){
            bias   [i]->values[n] += -(bias   [i]->getGradient(0)->get(n) * alpha);
        }
    }
}

void nn::Adam::initVectors(int size, nn::Data** weights, nn::Data** bias) {
    for(int i = 0; i < size; i++){
        weights_fmv.emplace_back(weights[i]->size,0);
        weights_smv.emplace_back(weights[i]->size,0);
        bias_fmv   .emplace_back(   bias[i]->size,0);
        bias_smv   .emplace_back(   bias[i]->size,0);
    }
}
void nn::Adam::optimise(int size, nn::Data** weights, nn::Data** bias) {
    
    if(weights_fmv.empty()){
        initVectors(size, weights, bias);
    }
    
    timeStep++;
    for(int i=0; i < size;i++){
        Data* w =  weights    [i];
        Data* m = &weights_fmv[i];
        Data* v = &weights_smv[i];
        for(int k = 0; k < 2; k++){
            
            if(k == 1){
                w =  bias    [i];
                m = &bias_fmv[i];
                v = &bias_smv[i];
            }
            
            for(int j =0; j< w->size;j++){
                double gradient = w->getGradient(0)->get(j);
                (*m)(j) = (*m)(j)*beta1 + (1-beta1)*gradient;
                (*v)(j) = (*v)(j)*beta2 + (1-beta2)*(gradient*gradient);
                double m_corrected = (*m)(j)/(1-(pow(beta1,timeStep)));
                double v_corrected = (*v)(j)/(1-(pow(beta2,timeStep)));
                (*w)(j) += -(alpha*m_corrected)/(sqrt(v_corrected)+epsilon);

            }
            
            
        }
        
    }
    
}
nn::Adam::Adam(double alpha, double beta1, double beta2, double epsilon)
    : alpha(alpha), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

#endif