//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_NETWORK_H
#define KOIVISTO_NETWORK_H

#include "Layer.h"
#include "config.h"
#include "affine.h"
#include "structure.h"

namespace nn{

struct Network{
    
    const Layer* layers;
    
#ifdef NN_TRAIN
    Data** weights;
    Data** biases;
    Data*** outputs;
    Data*** activations;
#else
    Data** weights;
    Data** biases;
    Data** outputs;
    Data** activations;
#endif
    Network() {
        this->layers = nn::layers;
#ifdef NN_TRAIN
        this->weights = new Data*[LAYER_COUNT];
        this->biases = new Data*[LAYER_COUNT];
        this->outputs = new Data**[LAYER_COUNT];
        this->activations = new Data**[LAYER_COUNT];

        for(int i = 0; i < LAYER_COUNT; i++){
            this->weights[i] = new Data(layers[i].inputSize, layers[i].outputSize, NN_THREADS);
            this->biases[i] = new Data(layers[i].outputSize, NN_THREADS);
            float k = 1 / sqrt(layers[i].inputSize);
            this->weights[i]->randomise(-k, k);
    
            this->outputs[i] = new Data*[NN_THREADS];
            this->activations[i] = new Data*[NN_THREADS];
            for(int n = 0; n < NN_THREADS; n++){
                this->outputs[i][n] = new Data(layers[i].outputSize, 1);
                this->activations[i][n] = new Data(layers[i].outputSize, 1);
            }
        }
#else
        this->weights = new Data*[LAYER_COUNT];
        this->biases = new Data*[LAYER_COUNT];
        this->outputs = new Data*[LAYER_COUNT];
        this->activations = new Data*[LAYER_COUNT];

        for(int i = 0; i < LAYER_COUNT; i++){
            this->weights[i] = new Data(layers[i].inputSize, layers[i].outputSize);
            this->biases[i] = new Data(layers[i].inputSize, layers[i].outputSize);
            
            this->outputs[i] = new Data(NN_THREADS);
            this->activations[i] = new Data(NN_THREADS);
        }
#endif
        
        if(nn::loadWeights) {
            this->loadWeights(nn::weightsFile);
        }
        
    
    
    }

#ifdef NN_TRAIN
    void compute(Sample *sample, int threadID);
    void backprop(Sample *sample, int threadID);
    float getOutput(int threadID, int id=0);
    float applyLoss(Data* target, int threadID);
    void optimise();
    void mergeGrad();
    void clearGrad();
#else
    void compute(Sample *sample);
    float getOutput(int id=0){
        return this->activations[LAYER_COUNT-1]->values[id];
    }
#endif
    void  loadWeights(const std::string& weights);
    void writeWeights(const std::string& weights);


};

}

#endif    // KOIVISTO_NETWORK_H
