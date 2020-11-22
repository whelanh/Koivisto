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

#include "Data.h"

#include <cstring>

#ifdef NN_TRAIN
nn::Data::Data(int width, int createGradients) {
    this->width  = width;
    this->height = 1;
    this->size   = width;
    
    
    this->values = new (std::align_val_t(64)) float[size] {};

    if (createGradients) {
        this->gradient = new (std::align_val_t(64)) nn::Data*[createGradients];
        for(int i = 0; i < createGradients; i++){
            this->gradient[i] = new (std::align_val_t(64)) nn::Data(this->width, this->height, false);
        }
    }
}
nn::Data::Data(int width, int height, int createGradients) {
    this->width  = width;
    this->height = height;
    this->size   = width * height;
    
    
    this->values = new (std::align_val_t(64)) float[size] {};
    if (createGradients) {
        this->gradient = new (std::align_val_t(64)) nn::Data*[createGradients];
        for(int i = 0; i < createGradients; i++){
            this->gradient[i] = new (std::align_val_t(64)) nn::Data(this->width, this->height, false);
        }

    }
}
#else
nn::Data::Data(int width) {
    this->width  = width;
    this->height = 1;
    this->size   = width;
    
    assert(this->size % 8 == 0);
    
    this->values = new (std::align_val_t(64)) float[size] {};
}
nn::Data::Data(int width, int height) {
    this->width  = width;
    this->height = height;
    this->size   = width * height;
    
    assert(this->width % 8 == 0);
    assert(this->height % 8 == 0 || this->height == 1);
    
    this->values = new (std::align_val_t(64)) float[size] {};
}
#endif

nn::Data::~Data() {
    _mm_free(values);
#ifdef NN_TRAIN
    if (gradient != nullptr) {
        delete gradient;
    }
#endif
}

#ifdef NN_TRAIN
bool   nn::Data::hasGradient()                           { return gradient != nullptr; }
nn::Data*  nn::Data::getGradient(int id)      const {
    return gradient[id];
}
#endif

void   nn::Data::randomise(float lower, float upper) {
    for (int i = 0; i < size; i++) {
        this->values[i] = static_cast<float>(rand()) / RAND_MAX * (upper - lower) + lower;
    }
}
void   nn::Data::mergeInto(Data* other) {
    // adds the content of this data object into the other data object.
    for(int i = 0; i < size; i+= 8){
        // load our values and the target values into the register
        __m256 other_values = _mm256_load_ps(&other->values[i]);
        __m256   our_values = _mm256_load_ps(& this->values[i]);
        // stores the sum of our and their values inside the other data object.
        _mm256_store_ps(&other->values[i], _mm256_add_ps(other_values, our_values));
    }
}
void   nn::Data::clear(){
    // clears the content of this data object
    std::memset(this->values, 0, this->size * sizeof(float));
//    for(int i = 0; i < size; i++){
//        this->values[i] = 0;
//    }
}
float  nn::Data::get(int width)                    const { return values[width]; }
float& nn::Data::get(int width)                          { return values[width]; }
float  nn::Data::get(int width, int height)        const { return values[width + height * this->width]; }
float& nn::Data::get(int width, int height)              { return values[width + height * this->width]; }
float  nn::Data::operator()(int width)             const { return get(width); }
float& nn::Data::operator()(int width)                   { return get(width); }
float  nn::Data::operator()(int width, int height) const { return get(width, height); }
float& nn::Data::operator()(int width, int height)       { return get(width, height); }
float* nn::Data::getValues()                       const { return values; }
int    nn::Data::getWidth()                        const { return width; }
int    nn::Data::getHeight()                       const { return height; }
int    nn::Data::getSize()                         const { return size; }

