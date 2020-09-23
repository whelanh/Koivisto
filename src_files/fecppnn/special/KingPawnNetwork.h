//
// Created by finne on 9/22/2020.
//

#ifndef KOIVISTO_KINGPAWNMAPPER_H
#define KOIVISTO_KINGPAWNMAPPER_H

#endif    // KOIVISTO_KINGPAWNMAPPER_H

#include "../../Board.h"
#include "../layer/DenseLayer_Sparse_NF.h"

#define KING_INDEX(color, square) color * 64 + square
#define PAWN_INDEX(color, square) 128 + color * 48 + square - 8

namespace fecppnn {

class KingPawnNetwork {

    private:
    Network*              network;
    DenseLayer_Sparse_NF* inputLayer;

    public:
    KingPawnNetwork(Network* network) : network(network) {
        inputLayer = dynamic_cast<DenseLayer_Sparse_NF*>(network->getLayer(0));
    }
    void resetInput(Board* board) {
        
        inputLayer->clearInput();
        
        U64 bb = board->getPieces(WHITE, KING);
        while (bb) {
            inputLayer->adjustInput(KING_INDEX(WHITE, bitscanForward(bb)), 1);
            bb = lsbReset(bb);
        }
        bb = board->getPieces(BLACK, KING);
        while (bb) {
            inputLayer->adjustInput(KING_INDEX(BLACK, bitscanForward(bb)), 1);
            bb = lsbReset(bb);
        }

        bb = board->getPieces(WHITE, PAWN);
        while (bb) {
            inputLayer->adjustInput(PAWN_INDEX(WHITE, bitscanForward(bb)), 1);
            bb = lsbReset(bb);
        }
        bb = board->getPieces(BLACK, PAWN);
        while (bb) {
            inputLayer->adjustInput(PAWN_INDEX(BLACK, bitscanForward(bb)), 1);
            bb = lsbReset(bb);
        }
        
    }

    bool validateInput(Board* board) {
        U64 bb = board->getPieces(WHITE, KING);
        while (bb) {
            int index = KING_INDEX(WHITE, bitscanForward(bb));
            if (inputLayer->getInput()->get(index) != 1)
                return false;
            bb = lsbReset(bb);
        }
        bb = board->getPieces(BLACK, KING);
        while (bb) {
            int index = KING_INDEX(BLACK, bitscanForward(bb));
            if (inputLayer->getInput()->get(index) != 1)
                return false;
            bb = lsbReset(bb);
        }

        bb = board->getPieces(WHITE, PAWN);
        while (bb) {
            int index = PAWN_INDEX(WHITE, bitscanForward(bb));
            if (inputLayer->getInput()->get(index) != 1)
                return false;
            bb = lsbReset(bb);
        }
        bb = board->getPieces(BLACK, PAWN);
        while (bb) {
            int index = PAWN_INDEX(BLACK, bitscanForward(bb));

            if (inputLayer->getInput()->get(index) != 1)
                return false;

            bb = lsbReset(bb);
        }
        return true;
    }

    void printInputs() {
        
        for(int i = 0; i < inputLayer->getInputTracker().count(); i++){
            std::cout << inputLayer->getInputTracker().at(i) << std::endl;
        }
        
        printArrayBinary(inputLayer->getInput()->getValues(), 4*64 - 4 * 8);
    }

    double compute(){
        fecppnn::Data* g = network->compute();
        return g->get(0);
    }
    
    void onMove(Move m) {
        Square sqFrom = getSquareFrom(m);
        Square sqTo   = getSquareTo(m);
        Piece  pFrom  = getMovingPiece(m);
        Type   mType  = getType(m);
        Color  us     = pFrom / 6;
        Color  them   = 1 - us;
        int    factor = us == WHITE ? 1 : -1;

        if (isCapture(m) && !isEnPassant(m)) {
            Piece capturedPiece = getCapturedPiece(m);
            if (capturedPiece % 6 == PAWN) {
                inputLayer->adjustInput(PAWN_INDEX(them, sqTo), 0);
            }
        }

        if (pFrom % 6 == PAWN) {

            inputLayer->adjustInput(PAWN_INDEX(us, sqFrom), 0);
            if (!isPromotion(m))
                inputLayer->adjustInput(PAWN_INDEX(us, sqTo), 1);

            if (mType == EN_PASSANT) {
                inputLayer->adjustInput(PAWN_INDEX(them, sqTo - 8 * factor), 0);
            }
        } else if (pFrom % 6 == KING) {
            inputLayer->adjustInput(KING_INDEX(us, sqFrom), 0);
            inputLayer->adjustInput(KING_INDEX(us, sqTo), 1);
        }
    }

    void onUndoMove(Move m) {

        Square sqFrom = getSquareFrom(m);
        Square sqTo   = getSquareTo(m);
        Piece  pFrom  = getMovingPiece(m);
        Type   mType  = getType(m);
        Color  us     = pFrom / 6;
        Color  them   = 1 - us;
        int    factor = us == WHITE ? 1 : -1;

        if (isCapture(m) && !isEnPassant(m)) {
            Piece capturedPiece = getCapturedPiece(m);
            if (capturedPiece % 6 == PAWN) {
                inputLayer->adjustInput(PAWN_INDEX(them, sqTo), 1);
            }
        }

        if (pFrom % 6 == PAWN) {
            inputLayer->adjustInput(PAWN_INDEX(us, sqFrom), 1);
            if (!isPromotion(m))
                inputLayer->adjustInput(PAWN_INDEX(us, sqTo), 0);

            if (mType == EN_PASSANT) {
                inputLayer->adjustInput(PAWN_INDEX(them, sqTo - 8 * factor), 1);
            }
        } else if (pFrom % 6 == KING) {
            inputLayer->adjustInput(KING_INDEX(us, sqFrom), 1);
            inputLayer->adjustInput(KING_INDEX(us, sqTo), 0);
        }
    }

    Network*              getNetwork() const { return network; }
    void                  setNetwork(Network* network) { KingPawnNetwork::network = network; }
    DenseLayer_Sparse_NF* getInputLayer() const { return inputLayer; }
    void                  setInputLayer(DenseLayer_Sparse_NF* inputLayer) { KingPawnNetwork::inputLayer = inputLayer; }
};

}    // namespace fecppnn
