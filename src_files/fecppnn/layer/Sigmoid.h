/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_SIGMOID_H
#define KOIVISTO_SIGMOID_H

#include "Layer.h"

#include <cmath>
namespace fecppnn {
class Sigmoid : public Layer {
    public:
    Sigmoid(Layer* prevLayer) : Layer(prevLayer->getOutput()->getSize()) { this->connect(prevLayer); };

    void compute() override;
    void backprop() override;

    void              collectOptimisableData(std::vector<Data*>& vec) override;
    const std::string name() override;
};
}    // namespace fecppnn

#endif    // KOIVISTO_RELU_H
