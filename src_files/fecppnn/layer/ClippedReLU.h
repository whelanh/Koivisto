/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_CLIPPEDRELU_H
#define KOIVISTO_CLIPPEDRELU_H

#include "Layer.h"

#include <immintrin.h>

namespace fecppnn {
class ClippedReLU : public Layer {

    public:
    ClippedReLU(Layer* prevLayer);
    ;

    void compute() override;

    void backprop() override;

    void              collectOptimisableData(std::vector<Data*>& vec) override {}
    const std::string name() override;
};
}    // namespace fecppnn
#endif    // KOIVISTO_CLIPPEDRELU_H
