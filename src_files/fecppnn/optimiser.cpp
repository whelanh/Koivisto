//
// Created by Luecx on 21.11.2020.
//

#include "optimiser.h"

#include <iostream>

#ifdef NN_TRAIN

nn::Optimiser::Optimiser(double alpha) : alpha(alpha) {}
void nn::Optimiser::optimise(nn::Data** weights, nn::Data** bias) {
    for(int i = 0; i < LAYER_COUNT; i++){
        for(int n = 0; n < weights[i]->size; n++){
            weights[i]->values[n] += -(weights[i]->getGradient(0)->get(n) * alpha);
        }
        for(int n = 0; n < bias[i]->size; n++){
            bias   [i]->values[n] += -(bias   [i]->getGradient(0)->get(n) * alpha);
        }
    }
}

void nn::Adam::initVectors() {
    for(int i = 0; i < LAYER_COUNT; i++){
        weights_fmv[i] = new Data(layers[i].inputSize * layers[i].outputSize,0);
        weights_smv[i] = new Data(layers[i].inputSize * layers[i].outputSize,0);
        bias_fmv[i] = new Data(                      layers[i].outputSize,0);
        bias_smv[i] = new Data(                      layers[i].outputSize,0);
    }
}
void nn::Adam::optimise(nn::Data** weights, nn::Data** bias) {
    
    if(weights_fmv[0] == nullptr){
        initVectors();
    }
    
    timeStep++;
    for(int i=0; i < LAYER_COUNT;i++){
        Data* w = weights    [i];
        Data* m = weights_fmv[i];
        Data* v = weights_smv[i];
        for(int k = 0; k < 2; k++){
            
            if(k == 1){
                w = bias    [i];
                m = bias_fmv[i];
                v = bias_smv[i];
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

#endif