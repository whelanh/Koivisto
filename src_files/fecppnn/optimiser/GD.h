/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_GD_H
#define KOIVISTO_GD_H

#include "Optimiser.h"

#include <immintrin.h>
#include <iostream>
#include <memory.h>

namespace fecppnn {

class GD : public Optimiser {

    private:
    float              eta;
    std::vector<Data*> weights;

    public:
    GD(float p_eta) : eta(p_eta) {}

    void init(std::vector<Data*>& p_weights) override { this->weights = p_weights; }

    void update() override {
        __m256 lr = _mm256_set1_ps(-eta);

        for (Data* d : weights) {

            float* data = d->getValues();
            float* grad = d->getGradient()->getValues();

            int size = d->getSize();

            for (int i = 0; i < size; i += 8) {
                __m256 grads = _mm256_load_ps(&grad[i]);
                __m256 vals  = _mm256_load_ps(&data[i]);
                _mm256_store_ps(&data[i], _mm256_add_ps(vals, _mm256_mul_ps(grads, lr)));
            }

            //            for(int i = 0; i < size; i++){
            //                data[i] -= eta * grad[i];
            //            }

            memset(grad, 0, sizeof(float) * size);
        }
    }

    virtual ~GD() {}

    float getEta() const { return eta; }
    void  setEta(float p_eta) { GD::eta = p_eta; }
};

}    // namespace fecppnn
#endif    // KOIVISTO_GD_H
