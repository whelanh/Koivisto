//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_CONFIG_H
#define KOIVISTO_CONFIG_H

namespace nn{
    class   Data;
    struct  Optimiser;
}

typedef void  (*Activation) (nn::Data* output, const nn::Data* activation);
typedef void  (*BackProp)   (nn::Data* output, const nn::Data* activation);
typedef float (*Loss)       (nn::Data* output, const nn::Data* target);


#define NN_TRAIN
#ifndef NN_TRAIN
#define NN_RUN
#endif

#define NN_THREADS 1
#define NN_BATCH_SIZE 4096




#endif    // KOIVISTO_CONFIG_H
