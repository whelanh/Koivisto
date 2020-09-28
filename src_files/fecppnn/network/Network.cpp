//
// Created by finne on 9/14/2020.
//

#include "Network.h"

fecppnn::Loss*      fecppnn::Network::getLoss() const { return loss; }
void                fecppnn::Network::setLoss(Loss* loss) { Network::loss = loss; }
fecppnn::Optimiser* fecppnn::Network::getOptimiser() const { return optimiser; }
void                fecppnn::Network::setOptimiser(Optimiser* optimiser) { Network::optimiser = optimiser; }
fecppnn::Layer*     fecppnn::Network::getLayer(int index) { return layers.at(index); }
void                fecppnn::Network::updateWeights() { optimiser->update(); }
double              fecppnn::Network::backprop(Data* target) {

    double l = loss->computeLoss(layers.at(layers.size() - 1)->getOutput(), target);

    for (int i = layers.size() - 1; i >= 1; i--) {
        layers.at(i)->backprop();
    }

    return l;
}
double fecppnn::Network::backpropWithoutLossFunction() {
    for (int i = layers.size() - 1; i >= 1; i--) {
        layers.at(i)->backprop();
    }
    return 0;
}
fecppnn::Data* fecppnn::Network::compute() {

    for (Layer* l : layers) {
        l->compute();
    }
    return layers.at(layers.size() - 1)->getOutput();
}
void fecppnn::Network::addLayer(Layer* layer) {
    // add the layer to all the layers
    layers.push_back(layer);

    // keep track of all weights
    layer->collectOptimisableData(weights);
}
void fecppnn::Network::prepareTraining() {
    if (optimiser == nullptr) {
        std::cerr << "no optimiser is set for the network." << std::endl;
    }
    if (loss == nullptr) {
        std::cerr << "no loss is set for the network." << std::endl;
    }

    optimiser->init(weights);
}
void fecppnn::Network::loadWeights(const std::string& path) {
    FILE* infile = fopen(path.c_str(), "rb");

    for (Data* d : weights) {
        fread(d->getValues(), sizeof(float), d->getSize(), infile);
    }
    
    fclose(infile);
}
void fecppnn::Network::writeWeights(const std::string& path) {
    FILE* outfile = fopen(path.c_str(), "wb");

    for (Data* d : weights) {
        fwrite(d->getValues(), sizeof(float), d->getSize(), outfile);
    }

    fclose(outfile);
}
void fecppnn::Network::writeNetworkStructure(const std::string& path) {
    std::ofstream myfile;
    myfile.open(path);

    for (int i = 0; i < layers.size(); i++) {
        Layer* layer = layers.at(i);

        std::string layerName = layer->name();

        if (layerName == "InputLayer") {
            myfile << "InputLayer " << layer->getOutput()->getSize() << "\n";
        } else if (layerName == "DenseLayer_Sparse_NF") {
            myfile << "DenseLayer_Sparse_NF " << layer->getOutput()->getSize() << " "
                   << dynamic_cast<DenseLayer_Sparse_NF*>(layer)->getInput()->getSize() << " "
                   << "\n";
        } else if (layerName == "Concat") {

            auto it1 = std::find(layers.begin(), layers.end(), layer->getPreviousLayers().at(0));
            auto it2 = std::find(layers.begin(), layers.end(), layer->getPreviousLayers().at(1));

            int id1 = std::distance(layers.begin(), it1);
            int id2 = std::distance(layers.begin(), it2);

            myfile << "Concat " << layer->getOutput()->getSize() << " " << id1 << " " << id2 << "\n";
        } else if (layerName == "DenseLayer") {
            auto it1 = std::find(layers.begin(), layers.end(), layer->getPreviousLayers().at(0));
            int  id1 = std::distance(layers.begin(), it1);
            myfile << "DenseLayer " << layer->getOutput()->getSize() << " " << id1 << "\n";
        } else if (layerName == "ReLU") {
            auto it1 = std::find(layers.begin(), layers.end(), layer->getPreviousLayers().at(0));
            int  id1 = std::distance(layers.begin(), it1);
            myfile << "ReLU " << layer->getOutput()->getSize() << " " << id1 << "\n";
        } else if (layerName == "ClippedReLU") {
            auto it1 = std::find(layers.begin(), layers.end(), layer->getPreviousLayers().at(0));
            int  id1 = std::distance(layers.begin(), it1);
            myfile << "ClippedReLU " << layer->getOutput()->getSize() << " " << id1 << "\n";
        } else if (layerName == "Sigmoid") {
            auto it1 = std::find(layers.begin(), layers.end(), layer->getPreviousLayers().at(0));
            int  id1 = std::distance(layers.begin(), it1);
            myfile << "Sigmoid " << layer->getOutput()->getSize() << " " << id1 << "\n";
        } else {
            std::cout << "did not find this layer type" << std::endl;
        }
    }

    myfile.close();
}
