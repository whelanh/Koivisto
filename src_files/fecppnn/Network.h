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

    int size;
    
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
        this->size = LAYER_COUNT;
        this->layers = nn::layers;
#ifdef NN_TRAIN
        this->weights = new Data*[size];
        this->biases = new Data*[size];
        this->outputs = new Data**[size];
        this->activations = new Data**[size];

        for(int i = 0; i < size; i++){
            this->weights[i] = new Data(layers[i].inputSize, layers[i].outputSize, NN_THREADS);
            this->biases[i] = new Data(layers[i].inputSize, layers[i].outputSize, NN_THREADS);
            float k = 1 / sqrt(layers[i].inputSize);
            this->weights[i]->randomise(-k, k);
            this->biases[i]->randomise(-k, k);
    
            this->outputs[i] = new Data*[NN_THREADS];
            this->activations[i] = new Data*[NN_THREADS];
            for(int n = 0; n < NN_THREADS; n++){
                this->outputs[i][n] = new Data(layers[i].outputSize, 1);
                this->activations[i][n] = new Data(layers[i].outputSize, 1);
            }
        }
#else
        this->weights = new Data*[size];
        this->biases = new Data*[size];
        this->outputs = new Data*[size];
        this->activations = new Data*[size];

        for(int i = 0; i < size; i++){
            this->weights[i] = new Data(layers[i].inputSize, layers[i].outputSize);
            this->biases[i] = new Data(layers[i].inputSize, layers[i].outputSize);
            
            this->outputs[i] = new Data(NN_THREADS);
            this->activations[i] = new Data(NN_THREADS);
        }
#endif
    
    }

#ifdef NN_TRAIN
    void compute(Sample *sample, int threadID);
    void backprop(Sample *sample, int threadID);
    float getOutput(int threadID, int id=0){
        return this->activations[size-1][threadID]->values[id];
    }
    void mergeGrad();
    void clearGrad();
#else
    void compute(Sample *sample);
    float getOutput(int id=0){
        return this->activations[size-1]->values[id];
    }
#endif


};

}

#endif    // KOIVISTO_NETWORK_H
