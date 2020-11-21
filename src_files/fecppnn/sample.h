//
// Created by Luecx on 21.11.2020.
//

#ifndef KOIVISTO_SAMPLE_H
#define KOIVISTO_SAMPLE_H

#include <cstdint>

namespace nn{

struct Sample{
    std::vector<uint16_t> indices {};
};

}


#endif    // KOIVISTO_SAMPLE_H
