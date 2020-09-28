/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_CONCAT_H
#define KOIVISTO_CONCAT_H

#include "Layer.h"
namespace fecppnn {
class Concat : public Layer {

    bool flipInputs = false;

    public:
    Concat(Layer* prev1, Layer* prev2);

    void compute() override;
    void backprop() override;

    void              collectOptimisableData(std::vector<Data*>& vec) override {}
    const std::string name() override;

    bool isFlipInputs() const;
    void setFlipInputs(bool flipInputs);
};
}    // namespace fecppnn

#endif    // KOIVISTO_CONCAT_H
