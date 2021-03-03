//
// Created by Luecx on 28.02.2021.
//

#ifndef KOIVISTO_NNUE_EVAL_H
#define KOIVISTO_NNUE_EVAL_H

#include <immintrin.h>

namespace nnue{

constexpr int IN_SIZE = 10 * 64 * 64;
constexpr int L1_SIZE = 256;
constexpr int L2_SIZE = 32;
constexpr int L3_SIZE = 32;
constexpr int L4_SIZE = 1;

alignas(32) uint16_t WHITE_INPUT[L1_SIZE];
alignas(32) uint16_t BLACK_INPUT[L1_SIZE];
alignas(32) uint16_t L1_OUTPUT[2*L1_SIZE];
alignas(32) uint16_t L2_OUTPUT[L2_SIZE];
alignas(32) uint16_t L3_OUTPUT[L3_SIZE];
alignas(32) uint16_t L4_OUTPUT[L4_SIZE];

template<int I, int O>
void input_set(int16_t *inputs, int16_t *outputs, int16_t* weights, uint16_t index, bool value){
    // dont change anything if the input value is already set
    if(value == inputs[index]) return;
    
    const     int offset = O * index;
    constexpr int rows   = O / 16;
    
    inputs[index] = value;
    
    __m256i* wgt = (__m256i*)(&weights[offset]);
    __m256i* out = (__m256i*)( outputs);
    
    if(value){
        for(int i = 0; i < rows; i++){
            out[i] = _mm256_add_epi16(out[i],wgt[i]);
        }
    }else{
        for(int i = 0; i < rows; i++){
            out[i] = _mm256_sub_epi16(out[i],wgt[i]);
        }
    }
}

template<int I, int O>
void affine(int16_t* mat, uint16_t* in, int32_t* bias, uint32_t* out){

}


void test(){
    auto *l1_inputs     = ( int16_t*) _mm_malloc(      10 * 64 * 64 * sizeof( int16_t), 32);
    auto *l1_weights    = ( int16_t*) _mm_malloc(256 * 10 * 64 * 64 * sizeof( int16_t), 32);
    auto *l2_inputs     = ( int16_t*) _mm_malloc(256                * sizeof( int16_t), 32);
    
    std::memset(l1_inputs , 0,       10 * 64 * 64 * sizeof( int16_t));
    std::memset(l1_weights, 0, 256 * 10 * 64 * 64 * sizeof( int16_t));
    std::memset(l2_inputs , 0, 256 *                sizeof( int16_t));
    
    for(int i = 0; i < 256 * 10 * 64 * 64; i++){
        l1_weights[i] = (i * 7) % 12;
    }
    
    input_set<10 * 64 * 64, 256>(l1_inputs, l2_inputs, l1_weights, 3, 1);
    for(int i = 0; i < 256; i++){
        std::cout << l2_inputs[i] << std::endl;
    }
    input_set<10 * 64 * 64, 256>(l1_inputs, l2_inputs, l1_weights, 3, 0);
    for(int i = 0; i < 256; i++){
        std::cout << l2_inputs[i] << std::endl;
    }
    
    
}



}

#endif    // KOIVISTO_NNUE_EVAL_H
