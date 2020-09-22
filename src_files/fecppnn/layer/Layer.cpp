/***********************************************************************************
 * Copyright (C) 2020-2021 {Finn Eggers} <{mail@finneggers.de}>                    *
 *                                                                                 *
 * This file is part of fecppnn.                                                   *
 *                                                                                 *
 * fecppnn can not be copied and/or distributed without the express                *
 * permission of Finn Eggers                                                       *
 ***********************************************************************************/

#include "Layer.h"
fecppnn::Layer::Layer(int size) { this->output = new Data(size, true); }

void fecppnn::Layer::connect(Layer* prevLayer) {
    if (prevLayer->nextLayers.size() != 0) {
        std::cerr << "spreading data to different layers is not important in this version for performance reasons"
                  << std::endl;
        exit(-1);
    }

    previousLayers.push_back(prevLayer);
    prevLayer->nextLayers.push_back(this);
}
void                       fecppnn::Layer::setOutput(Data* p_output) { Layer::output = p_output; }
fecppnn::Data*                      fecppnn::Layer::getOutput() const { return output; }
fecppnn::Data*                      fecppnn::Layer::getInput() { return previousLayers[0]->getOutput(); }
fecppnn::Data*                      fecppnn::Layer::getInput(int index) { return previousLayers[index]->getOutput(); }
const std::vector<fecppnn::Layer*>& fecppnn::Layer::getNextLayers() const { return nextLayers; }
const std::vector<fecppnn::Layer*>& fecppnn::Layer::getPreviousLayers() const { return previousLayers; }
