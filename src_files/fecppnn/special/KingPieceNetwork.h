//
// Created by finne on 9/22/2020.
//

#ifndef KOIVISTO_KINGPIECENETWORK_H
#define KOIVISTO_KINGPIECENETWORK_H

#include "../../Board.h"
#include "../layer/DenseLayer_Sparse_NF.h"
#include "../network/Network.h"

#define WHITE_INDEX(kingPos, piece, square) kingPos * 10 * 64 + piece * 64 + square
#define BLACK_INDEX(kingPos, piece, square) mirrorSquare(kingPos) * 10 * 64 + piece * 64 + mirrorSquare(square)

namespace fecppnn {

class KingPieceNetwork {

    private:
    Network*              network;
    DenseLayer_Sparse_NF* whiteInput;
    DenseLayer_Sparse_NF* blackInput;
    Concat*               concat;

    public:
    KingPieceNetwork(Network* network) : network(network) {
        whiteInput = dynamic_cast<DenseLayer_Sparse_NF*>(network->getLayer(0));
        blackInput = dynamic_cast<DenseLayer_Sparse_NF*>(network->getLayer(1));
        concat     = dynamic_cast<Concat*>(network->getLayer(2));
    }
    void resetInput(Board* board) {
        
        whiteInput->clearInput();
        blackInput->clearInput();
        Square wKingSq = bitscanForward(board->getPieces(WHITE, KING));
        Square bKingSq = bitscanForward(board->getPieces(BLACK, KING));
        
        for (Piece p = WHITE_PAWN; p < BLACK_KING; p++) {

            if (p % 6 == KING)
                continue;

            U64 bb = board->getPieces()[p];
            while (bb) {
                Square s = bitscanForward(bb);
                whiteInput->adjustInput(WHITE_INDEX(wKingSq, p % 6, s), 1);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, p % 6, s), 1);
                bb = lsbReset(bb);
            }
        }
        concat->setFlipInputs(board->getActivePlayer() == BLACK);
    }

    bool validateInput(Board* board) {
        //        U64 bb = board->getPieces(WHITE, KING);
        //        while (bb) {
        //            int index = KING_INDEX(WHITE, bitscanForward(bb));
        //            if (inputLayer->getInput()->get(index) != 1)
        //                return false;
        //            bb = lsbReset(bb);
        //        }
        //        bb = board->getPieces(BLACK, KING);
        //        while (bb) {
        //            int index = KING_INDEX(BLACK, bitscanForward(bb));
        //            if (inputLayer->getInput()->get(index) != 1)
        //                return false;
        //            bb = lsbReset(bb);
        //        }
        //
        //        bb = board->getPieces(WHITE, PAWN);
        //        while (bb) {
        //            int index = PAWN_INDEX(WHITE, bitscanForward(bb));
        //            if (inputLayer->getInput()->get(index) != 1)
        //                return false;
        //            bb = lsbReset(bb);
        //        }
        //        bb = board->getPieces(BLACK, PAWN);
        //        while (bb) {
        //            int index = PAWN_INDEX(BLACK, bitscanForward(bb));
        //
        //            if (inputLayer->getInput()->get(index) != 1)
        //                return false;
        //
        //            bb = lsbReset(bb);
        //        }
        //        return true;
    }

    void printInputs() {

        //        for(int i = 0; i < inputLayer->getInputTracker().count(); i++){
        //            std::cout << inputLayer->getInputTracker().at(i) << std::endl;
        //        }
        //
        //        printArrayBinary(inputLayer->getInput()->getValues(), 4*64 - 4 * 8);
    }

    double compute() {
        fecppnn::Data* g = network->compute();
        return g->get(0);
    }

    void onMove(Move m) {
        //        Square sqFrom = getSquareFrom(m);
        //        Square sqTo   = getSquareTo(m);
        //        Piece  pFrom  = getMovingPiece(m);
        //        Type   mType  = getType(m);
        //        Color  us     = pFrom / 6;
        //        Color  them   = 1 - us;
        //        int    factor = us == WHITE ? 1 : -1;
        //
        //        if (isCapture(m) && !isEnPassant(m)) {
        //            Piece capturedPiece = getCapturedPiece(m);
        //            if (capturedPiece % 6 == PAWN) {
        //                inputLayer->adjustInput(PAWN_INDEX(them, sqTo), 0);
        //            }
        //        }
        //
        //        if (pFrom % 6 == PAWN) {
        //
        //            inputLayer->adjustInput(PAWN_INDEX(us, sqFrom), 0);
        //            if (!isPromotion(m))
        //                inputLayer->adjustInput(PAWN_INDEX(us, sqTo), 1);
        //
        //            if (mType == EN_PASSANT) {
        //                inputLayer->adjustInput(PAWN_INDEX(them, sqTo - 8 * factor), 0);
        //            }
        //        } else if (pFrom % 6 == KING) {
        //            inputLayer->adjustInput(KING_INDEX(us, sqFrom), 0);
        //            inputLayer->adjustInput(KING_INDEX(us, sqTo), 1);
        //        }
    }

    void onUndoMove(Move m) {

        //        Square sqFrom = getSquareFrom(m);
        //        Square sqTo   = getSquareTo(m);
        //        Piece  pFrom  = getMovingPiece(m);
        //        Type   mType  = getType(m);
        //        Color  us     = pFrom / 6;
        //        Color  them   = 1 - us;
        //        int    factor = us == WHITE ? 1 : -1;
        //
        //        if (isCapture(m) && !isEnPassant(m)) {
        //            Piece capturedPiece = getCapturedPiece(m);
        //            if (capturedPiece % 6 == PAWN) {
        //                inputLayer->adjustInput(PAWN_INDEX(them, sqTo), 1);
        //            }
        //        }
        //
        //        if (pFrom % 6 == PAWN) {
        //            inputLayer->adjustInput(PAWN_INDEX(us, sqFrom), 1);
        //            if (!isPromotion(m))
        //                inputLayer->adjustInput(PAWN_INDEX(us, sqTo), 0);
        //
        //            if (mType == EN_PASSANT) {
        //                inputLayer->adjustInput(PAWN_INDEX(them, sqTo - 8 * factor), 1);
        //            }
        //        } else if (pFrom % 6 == KING) {
        //            inputLayer->adjustInput(KING_INDEX(us, sqFrom), 1);
        //            inputLayer->adjustInput(KING_INDEX(us, sqTo), 0);
        //        }
    }

    Network* getNetwork() const { return network; }
    void     setNetwork(Network* network) { KingPieceNetwork::network = network; }
};

}    // namespace fecppnn

#endif