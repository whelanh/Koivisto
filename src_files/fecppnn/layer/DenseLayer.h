
/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_DENSELAYER_H
#define KOIVISTO_DENSELAYER_H

#include "Layer.h"

#include <immintrin.h>
#include <iostream>
#include <math.h>

namespace fecppnn {
class DenseLayer : public Layer {

    private:
    Data* weights;
    Data* bias;

    public:
    DenseLayer(Layer* previousLayer, int size);

    virtual ~DenseLayer() {
        delete weights;
        delete bias;
    }

    void compute() override;
    void backprop() override;
    void collectOptimisableData(std::vector<Data*>& vec) override {
        vec.push_back(weights);
        vec.push_back(bias);
    }
    const std::string name() override;

    Data* getWeights() const;
    void  setWeights(Data* p_weights);
    Data* getBias() const;
    void  setBias(Data* p_bias);
};
}    // namespace fecppnn

#endif    // KOIVISTO_DENSELAYER_H
