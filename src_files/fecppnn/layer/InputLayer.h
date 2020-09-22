/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_INPUTLAYER_H
#define KOIVISTO_INPUTLAYER_H

#include "Layer.h"
namespace fecppnn {
class InputLayer : public Layer {

    public:
    void compute() override {}
    void backprop() override {}
    void collectOptimisableData(std::vector<Data*>& vec) override {}

    const std::string name() override { return "InputLayer"; }

    InputLayer(int size) : Layer(size) {}
};
}    // namespace fecppnn
#endif    // KOIVISTO_INPUTLAYER_H
