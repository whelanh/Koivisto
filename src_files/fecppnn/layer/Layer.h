
/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_LAYER_H
#define KOIVISTO_LAYER_H

#include "../data/Data.h"

#include <iostream>
#include <vector>

namespace fecppnn {
class Layer {

    private:
    Data* output;

    std::vector<Layer*> previousLayers {};
    std::vector<Layer*> nextLayers {};

    public:
    virtual void              compute()                                       = 0;
    virtual void              backprop()                                      = 0;
    virtual void              collectOptimisableData(std::vector<Data*>& vec) = 0;
    virtual const std::string name()                                          = 0;

    Layer(int size);
    virtual ~Layer() { delete output; }

    Data* getInput(int index);
    Data* getInput();
    Data* getOutput() const;

    void setOutput(Data* p_output);

    void connect(Layer* prevLayer);

    [[nodiscard]] const std::vector<Layer*>& getPreviousLayers() const;
    [[nodiscard]] const std::vector<Layer*>& getNextLayers() const;
};
}    // namespace fecppnn
#endif    // KOIVISTO_LAYER_H
