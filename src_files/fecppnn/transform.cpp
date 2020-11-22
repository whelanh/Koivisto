//
// Created by Luecx on 21.11.2020.
//
#include "transform.h"
#if NN_TYPE == FULL || NN_TYPE == RELATIVE

// the transformation for FULL or RELATIVE will assume a completely dense input
// this means that we simply do a dense layer with a set of given indices.
// this is mainly used for training as during real games, the input should be updated incrementally
void nn::affine_transformation_input(Sample* in, nn::Data* weights, Data* bias, nn::Data* output) {
    
    // extract the output values to which we write the transformation
    // there is no input data object as the Sample contains all relevant information
    // Note that the sample only contains indices for the places where there is a "1"
    // -> there is only 0 or 1 as an input for a neuron
    // The transformation is similar to the affine transformation which applies: o = A*x + b
    // where A is the weights matrix, b is the bias and x is the input encoded in the sample.
    float* outputValues = output ->values;
    float* biasValues   = bias   ->values;
    float* weightValues = weights->values;
    
    // it makes sense to reset the output values to the bias first and later add the matrix-vector product
    for(int i = 0; i < output->size; i++){
        outputValues[i] = biasValues[i];
    }
    
    for(uint16_t &index:in->indices){
        
        // we can only do the chunks of 8 with avx instructions
        // the rest must be done manually
        int size = output->size;
        if(size % 8 != 0){
            size -= size % 8;
        }
        // we assume that the output size of the very first layer is always a multiple of 8!
        for(int n = 0; n < size; n+=8){
            // get the gradients into the register aswell as the output which we want to write to
            __m256 wvalues = _mm256_load_ps(&(weightValues[index * output->size + n]));
            __m256 ovalues = _mm256_load_ps(&(outputValues[                       n]));
            // add the element-wise multiplication of the weights. For this, add the weights for the activated
            // input neuron (output = 1) to the output
            _mm256_store_ps(&outputValues[n],_mm256_add_ps(ovalues, wvalues));
        }
        for(int n = size; n < output->size; n++){
            outputValues[n] += weightValues[index * output->size + n];
        }
    }
}
#ifdef NN_TRAIN
// the backpropagation of the transformation for FULL or RELATIVE will assume a completely dense input
// this means that we simply do a dense layer with a set of given indices.
// this is mainly used for training as during real games, the input should be updated incrementally
void nn::affine_transformation_input_backprop(Sample* in, nn::Data* weights, Data* bias, nn::Data* output, int threadID) {
    
    // extract the weight gradient values which we want to compute.
    // there is no input data object as the Sample contains all relevant information.
    // Note that the sample only contains indices for the places where there is a "1"
    // -> there is only 0 or 1 as an input for a neuron
    // The transformation is similar to the backpropagation of the affine transformation
    // which computes gradients for a weights connecting node i with node o by doing:
    // grad(w_io) += output(i) * grad(o)
    float* weightsGrad = weights->getGradient(threadID)->values;
    float* biasGrad    = bias   ->getGradient(threadID)->values;
    float* outputGrad  = output ->getGradient(0       )->values;
    
    // as the bias is simply added, it can be considered a weight with a standard output of 1.
    // So we only need to add the output gradient
    for(int i = 0; i < output->size; i++){
        biasGrad[i] += outputGrad[i];
    }
    
    // going through each index, applying the rules described above
    // Note that this assumes, as well as the forwar step, that the output size is a multiple of 8
    // Otherwise a SIGSEGV will occur as we try to load 256 bit into a register to which we dont have access.
    for(uint16_t &index:in->indices){
        // we can only do the chunks of 8 with avx instructions
        // the rest must be done manually
        int size = output->size;
        if(size % 8 != 0){
            size -= size % 8;
        }
        for(int n = 0; n < size; n+=8){
            // get the weight gradient which we want to increment as well as the output gradient
            __m256 wgrad = _mm256_load_ps(&(weightsGrad[index * output->size + n]));
            __m256 ograd = _mm256_load_ps(&( outputGrad[                       n]));

            _mm256_store_ps(&(weightsGrad[index * output->size + n]), _mm256_add_ps(wgrad, ograd));
        }
        for(int n = size; n < output->size; n++){
            weightsGrad[index * output->size + n] += outputGrad[n];
        }
    }
}
#endif
#else

#endif
