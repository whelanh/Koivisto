//
// Created by finne on 9/22/2020.
//

#ifndef KOIVISTO_DENSELAYER_SPARSE_NF_H
#define KOIVISTO_DENSELAYER_SPARSE_NF_H

#include "Layer.h"

#include <immintrin.h>
#include <memory.h>

#define NON_FULLY_UPDATED_RUNS 1000

namespace fecppnn {
struct InputTracker {
    private:
    int indices[64] {};

    int size = 0;

    public:
    void put(int index) { indices[size++] = index; }

    void remove(int index) {
        for (int i = 0; i < size; i++) {
            if (indices[i] == index) {
                indices[i] = indices[size - 1];
                size--;
                return;
            }
        }
    }

    bool contains(int index) {
        for (int i = 0; i < size; i++) {
            if (indices[i] == index) {
                return true;
            }
        }
        return false;
    }

    void clear() { size = 0; }

    int count() const { return size; }

    int& at(int index) { return indices[index]; }
};

class DenseLayer_Sparse_NF : public Layer {

    private:
    InputTracker inputTracker;
    Data*        weights;
    Data*        bias;
    Data*        input;
    Data*        intermediateOutput;

    int  skippedComputations         = 0;
    bool changedSinceLastComputation = false;

    void              compute() override;
    void              backprop() override;
    const std::string name() override;
    void              collectOptimisableData(std::vector<Data*>& vec) override {
        vec.push_back(weights);
        vec.push_back(bias);
    }

    public:
    DenseLayer_Sparse_NF(int inSize, int outSize) : Layer(outSize) {
        input   = new Data(inSize, true);
        weights = new Data(inSize * outSize, true);
        weights->randomise(-1, 1);
        bias = new Data(outSize, true);
        bias->randomise(-1, 1);
        
        intermediateOutput = new Data(outSize, true);
    }
    void adjustInput(int index, float val);
    void clearInput() {
        inputTracker.clear();
        memset(getInput()->getValues(), 0, sizeof(float) * getInput()->getSize());
        memset(getOutput()->getValues(), 0, sizeof(float) * getOutput()->getSize());
    }

    Data* getWeights() { return weights; }
    void  setWeights(Data* weights) { DenseLayer_Sparse_NF::weights = weights; }
    Data* getBias() { return bias; }
    void  setBias(Data* bias) { DenseLayer_Sparse_NF::bias = bias; }

    InputTracker& getInputTracker() ;
    Data*         getInput() const;
    int           getSkippedComputations() const;
    bool          isChangedSinceLastComputation() const;
};
}    // namespace fecppnn
#endif    // KOIVISTO_DENSELAYER_SPARSE_NF_H
