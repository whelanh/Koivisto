/****************************************************************************************************
 *                                                                                                  *
 *                                     Koivisto UCI Chess engine                                    *
 *                           by. Kim Kahre, Finn Eggers and Eugenio Bruno                           *
 *                                                                                                  *
 *                 Koivisto is free software: you can redistribute it and/or modify                 *
 *               it under the terms of the GNU General Public License as published by               *
 *                 the Free Software Foundation, either version 3 of the License, or                *
 *                                (at your option) any later version.                               *
 *                    Koivisto is distributed in the hope that it will be useful,                   *
 *                  but WITHOUT ANY WARRANTY; without even the implied warranty of                  *
 *                   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                  *
 *                           GNU General Public License for more details.                           *
 *                 You should have received a copy of the GNU General Public License                *
 *                 along with Koivisto.  If not, see <http://www.gnu.org/licenses/>.                *
 *                                                                                                  *
 ****************************************************************************************************/

#ifndef KOIVISTO_DATA_H
#define KOIVISTO_DATA_H


#include <cassert>
#include <iomanip>
#include <new>
#include <ostream>
#include "config.h"
#include <immintrin.h>

namespace nn {

class Data {
    public:
    
    int width = 0;
    int height = 0;
    int size = 0;
    int nGradients;
    float* values = nullptr;
#ifdef NN_TRAIN
    Data**  gradient = nullptr;
#endif

#ifdef NN_TRAIN
    Data(int width, int createGradients);
    Data(int width, int height, int createGradients);
    Data(Data &&data);
    Data(const Data &data);;
#else
    Data(int width);
    Data(int width, int height);
#endif
    virtual ~Data();

#ifdef NN_TRAIN
    bool hasGradient();
    Data* getGradient(int index) const;
#endif

    float  get(int width) const;
    float& get(int width);
    float  get(int width, int height) const;
    float& get(int width, int height);

    float  operator()(int width) const;
    float& operator()(int width);
    float  operator()(int width, int height) const;
    float& operator()(int width, int height);

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

    void randomise(float lower, float upper);
    void mergeInto(Data* other);
    void clear();
    
    float* getValues() const;

    int getWidth() const;
    int getHeight() const;
    int getSize() const;

};
}    // namespace nn

#endif    // KOIVISTO_DATA_H
