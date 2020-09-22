/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#ifndef KOIVISTO_DATA_H
#define KOIVISTO_DATA_H

#include <cassert>
#include <iomanip>
#include <new>
#include <ostream>

namespace fecppnn {
class Data {

    private:
    int width;
    int height;
    int size;

    float* values;
    Data*  gradient = nullptr;

    public:
    Data(int p_width, bool createGradients) {
        this->width  = p_width;
        this->height = 1;
        this->size   = p_width;

        assert(this->width % 8 == 0);

        this->values = new (std::align_val_t(32)) float[size] {};

        if (createGradients) {
            this->gradient = new Data(this->width, false);
        }
    }

    Data(int p_width, int p_height, bool createGradients) {
        this->width  = p_width;
        this->height = p_height;
        this->size   = p_width * p_height;

        assert(this->width % 8 == 0);
        assert(this->height % 8 == 0 || this->height == 1);

        this->values = new (std::align_val_t(32)) float[size] {};
        if (createGradients) {
            this->gradient = new Data(this->width, this->height, false);
        }
    }

    virtual ~Data() {
        //        delete[](std::align_val_t(32), values);
        if (gradient != nullptr) {
            delete gradient;
        }
    }

    bool hasGradient() { return gradient != nullptr; }

    float  get(int p_width) const { return values[p_width]; }
    float& get(int p_width) { return values[p_width]; }
    float  get(int p_width, int p_height) const { return values[p_width + p_height * this->width]; }
    float& get(int p_width, int p_height) { return values[p_width + p_height * this->width]; }

    float  operator()(int p_width) const { return get(p_width); }
    float& operator()(int p_width) { return get(p_width); }
    float  operator()(int p_width, int p_height) const { return get(p_width, p_height); }
    float& operator()(int p_width, int p_height) { return get(p_width, p_height); }

    friend std::ostream& operator<<(std::ostream& os, const Data& data) {

        os << std::fixed << std::setprecision(3);
        for (int i = 0; i < data.height; i++) {
            for (int n = 0; n < data.width; n++) {
                os << std::setw(10) << data(n, i);
            }
            os << "\n";
        }
        return os;
    }

    void randomise(float lower, float upper) {
        for (int i = 0; i < size; i++) {
            this->values[i] = static_cast<float>(rand()) / RAND_MAX * (upper - lower) + lower;
        }
    }

    float* getValues() const { return values; }

    [[nodiscard]] int getWidth() const { return width; }
    [[nodiscard]] int getHeight() const { return height; }
    [[nodiscard]] int getSize() const { return size; }

    Data* getGradient() const { return gradient; }
};
}    // namespace fecppnn

#endif    // KOIVISTO_DATA_H
