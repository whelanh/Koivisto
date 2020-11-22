//
// Created by Luecx on 21.11.2020.
//
#include "Network.h"

#ifdef NN_TRAIN
void nn::Network::compute(Sample* sample, int threadID) {

    affine_transformation_input(sample, weights[0], biases[0], outputs[0][threadID]);
    
    this->layers[0].activation(outputs[0][threadID], activations[0][threadID]);
    for(int i = 1; i < size; i++){
        affine_transformation(weights[i], biases[i], activations[i-1][threadID], outputs[i][threadID]);
        this->layers[i].activation(outputs[i][threadID], activations[i][threadID]);
    }
}
void nn::Network::backprop(Sample* sample, int threadID) {
    for(int i = size-1; i > 0; i--){
        this->layers[i].backprop(activations[i][threadID], outputs[i][threadID]);
        affine_transformation_backprop(weights[i], biases[i], activations[i-1][threadID], outputs[i][threadID], threadID);
    }
    this->layers[0].backprop(activations[0][threadID], outputs[0][threadID]);
    affine_transformation_input_backprop(sample, weights[0], biases[0], outputs[0][threadID], threadID);
}
void nn::Network::mergeGrad() {
    // merges the gradients of all the weights/biases into the first gradient entry
    // it does not delete the other gradients so optimally, clearGrad() should be called afterwards
    for(int i = 0; i < size; i++){
        for(int t = 1; t < NN_THREADS; t++){
            weights[i]->getGradient(t)->mergeInto(weights[i]->getGradient(0));
            biases [i]->getGradient(t)->mergeInto(biases [i]->getGradient(0));
        }
    }
}
void nn::Network::clearGrad() {
    // deletes the gradients for all weights/biases for all threads
    for (int i = 0; i < size; i++) {
        for (int t = 0; t < NN_THREADS; t++) {
            weights[i]->getGradient(t)->clear();
            biases [i]->getGradient(t)->clear();
        }
    }
}
#else
void compute(Sample* sample) {
    affine_transformation_input(sample, weights[0], biases[0], outputs[0]);
    this->layers[0].activation(outputs[0], activations[0]);
    for (int i = 1; i < size; i++) {
        affine_transformation(weights[i], biases[i], activations[i - 1], outputs[i]);
        this->layers[i].activation(outputs[i][threadID], activations[i]);
    }
}
#endif
