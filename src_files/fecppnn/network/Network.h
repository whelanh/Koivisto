//
// Created by finne on 9/14/2020.
//

#ifndef FECPPNN_NETWORK_H
#define FECPPNN_NETWORK_H

#include "../../Util.h"
#include "../layer/ClippedReLU.h"
#include "../layer/Concat.h"
#include "../layer/DenseLayer.h"
#include "../layer/DenseLayer_Sparse_NF.h"
#include "../layer/InputLayer.h"
#include "../layer/Layer.h"
#include "../layer/ReLU.h"
#include "../layer/Sigmoid.h"
#include "../loss/Loss.h"
#include "../optimiser/Optimiser.h"

#include <fstream>
#include <vector>

namespace fecppnn {

class Network {

    private:
    std::vector<Layer*> layers;
    std::vector<Data*>  weights;
    Loss*               loss      = nullptr;
    Optimiser*          optimiser = nullptr;

    public:
    // adds a new layer to the network
    void addLayer(Layer* layer);

    // prepares the loss to update the weights. only called once before training starts or a new epoch starts.
    void prepareTraining();

    // does the forward pass
    Data* compute();

    // does the backward pass and sums up the gradients for the weights
    double backpropWithoutLossFunction();

    // does the backward pass and sums up the gradients for the weights
    double backprop(Data* target);

    // updates the weights
    void updateWeights();

    // returns the layer at the given index
    Layer* getLayer(int index);

    // loads the weights as binary data from the given file
    void loadWeights(const std::string& path);

    // writes the weights as binary data to the given file
    void writeWeights(const std::string& path);

    // writes the network structure into a file which can be used to create a new network
    void writeNetworkStructure(const std::string& path);

    Loss*      getLoss() const;
    void       setLoss(Loss* loss);
    Optimiser* getOptimiser() const;
    void       setOptimiser(Optimiser* optimiser);
};

/**
 * creates a network from the given file which contains informations about the layers. this includes the layer type, its
 * size and the previous layer
 * @param file
 * @return
 */
static Network* createNetwork(const std::string& file) {

    std::ifstream infile(file);
    Network*      network = new Network();
    std::string   line;
    
    
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        
        std::vector<std::string> splits;
        splitString(line, splits, ' ');

        std::string id   = splits.at(0);
        int         size = stoi(splits.at(1));
        
        if (id == "InputLayer"){
            network->addLayer(new InputLayer(size));
            continue;
        }
        if (id == "DenseLayer_Sparse_NF"){
            network->addLayer(new DenseLayer_Sparse_NF(stoi(splits.at(2)), size));
            continue;
        }
        
        Layer* prevLayer = nullptr;
        if (splits.size() > 2) {
            prevLayer = network->getLayer(stoi(splits.at(2)));
        }
        
        if (id == "Concat")
            network->addLayer(new Concat(prevLayer, network->getLayer(stoi(splits.at(3)))));
        if (id == "DenseLayer")
            network->addLayer(new DenseLayer(prevLayer, size));
        if (id == "ReLU")
            network->addLayer(new ReLU(prevLayer));
        if (id == "ClippedReLU")
            network->addLayer(new ClippedReLU(prevLayer));
        if (id == "Sigmoid")
            network->addLayer(new Sigmoid(prevLayer));
    }

    return network;
}

}    // namespace fecppnn

#endif    // FECPPNN_NETWORK_H
