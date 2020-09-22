/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_OPTIMISER_H
#define KOIVISTO_OPTIMISER_H

#include "../data/Data.h"

#include <vector>
namespace fecppnn {
class Optimiser {

    public:
    virtual void init(std::vector<Data*>& weights) = 0;

    virtual void update() = 0;
};
}    // namespace fecppnn

#endif    // KOIVISTO_OPTIMISER_H
