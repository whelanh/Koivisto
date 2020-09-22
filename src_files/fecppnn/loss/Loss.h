/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_LOSS_H
#define KOIVISTO_LOSS_H

#include "../data/Data.h"

namespace fecppnn {

class Loss {

    public:
    virtual float computeLoss(Data* output, Data* expected) = 0;
};

}    // namespace fecppnn

#endif    // KOIVISTO_LOSS_H
