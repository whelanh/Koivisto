/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_MSE_H
#define KOIVISTO_MSE_H

#include "Loss.h"

namespace fecppnn {

class MSE : public Loss {
    public:
    float computeLoss(Data* output, Data* expected) override {

        float loss = 0;

        for (int i = 0; i < output->getSize(); i++) {
            float difference = output->getValues()[i] - expected->getValues()[i];

            output->getGradient()->getValues()[i] = 2 * difference;
            loss += difference * difference;
        }

        return loss / output->getSize();
    }
};

}    // namespace fecppnn

#endif    // KOIVISTO_MSE_H
