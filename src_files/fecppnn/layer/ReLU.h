/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_RELU_H
#define KOIVISTO_RELU_H

#include "Layer.h"

#include <immintrin.h>
namespace fecppnn {

/**
 *                   .   /
 *                   .  /
 *                   . /
 *                   ./
 *       -------------....................
 *                   .
 *                   .
 *
 */
class ReLU : public Layer {
    public:
    ReLU(Layer* prevLayer) : Layer(prevLayer->getOutput()->getSize()) { this->connect(prevLayer); };

    void compute() override;
    void backprop() override;

    void              collectOptimisableData(std::vector<Data*>& vec) override {}
    const std::string name() override;
};
}    // namespace fecppnn
#endif    // KOIVISTO_RELU_H
