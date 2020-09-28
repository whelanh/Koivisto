//
// Created by finne on 9/22/2020.
//

#ifndef KOIVISTO_KINGPIECENETWORK_H
#define KOIVISTO_KINGPIECENETWORK_H

#include "../../Board.h"
#include "../layer/DenseLayer_Sparse_NF.h"
#include "../network/Network.h"

#define WHITE_INDEX(kingPos, piece, square) kingPos * 10 * 64 + piece * 64 + square
#define BLACK_INDEX(kingPos, piece, square)                                                                            \
    mirrorSquare(kingPos) * 10 * 64 + (piece + (piece < 6) ? 6 : -6) * 64 + mirrorSquare(square)

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

    void setActivePlayer(Board* board) { concat->setFlipInputs(board->getActivePlayer() == BLACK); }

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

                //                std::cout << WHITE_INDEX(wKingSq, p, s) << " " << (int)wKingSq << " " << (int)p << " "
                //                << (int)s << std::endl;

                whiteInput->adjustInput(WHITE_INDEX(wKingSq, p, s), 1);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, p, s), 1);
                bb = lsbReset(bb);
            }
        }
        concat->setFlipInputs(board->getActivePlayer() == BLACK);
    }

    bool validateInput(Board* board) {

        Square wKingSq = bitscanForward(board->getPieces(WHITE, KING));
        Square bKingSq = bitscanForward(board->getPieces(BLACK, KING));

        for (Piece p = WHITE_PAWN; p < BLACK_KING; p++) {

            if (p % 6 == KING)
                continue;

            U64 bb = board->getPieces()[p];
            while (bb) {
                Square s = bitscanForward(bb);

                if (!whiteInput->getInputTracker().contains(WHITE_INDEX(wKingSq, p, s))) {
                    return false;
                }
                if (!blackInput->getInputTracker().contains(BLACK_INDEX(bKingSq, p, s))) {
                    return false;
                }

                bb = lsbReset(bb);
            }
        }

        if (whiteInput->getInputTracker().count() + 2 != bitCount(*board->getOccupied()))
            return false;
        if (blackInput->getInputTracker().count() + 2 != bitCount(*board->getOccupied()))
            return false;

        return true;
    }

    double compute() {
        fecppnn::Data* g = network->compute();
        return g->get(0);
    }

    void onMove(Board* b, Move m) {
        Square sqFrom = getSquareFrom(m);
        Square sqTo   = getSquareTo(m);
        Piece  pFrom  = getMovingPiece(m);
        Type   mType  = getType(m);
        Color  color  = pFrom / 6;
        int    factor = color == 0 ? 1 : -1;

        Square wKingSq = bitscanForward(b->getPieces(WHITE, KING));
        Square bKingSq = bitscanForward(b->getPieces(BLACK, KING));

        concat->setFlipInputs(color == BLACK);

        if (pFrom % 6 == PAWN) {

            if (isPromotion(m)) {

                // setting m_pieces
                // this->unsetPiece(sqFrom);
                whiteInput->adjustInput(WHITE_INDEX(wKingSq, pFrom, sqFrom), 0);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, pFrom, sqFrom), 0);

                // setting the piece at the target destination
                whiteInput->adjustInput(WHITE_INDEX(wKingSq, promotionPiece(m), sqTo), 1);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, promotionPiece(m), sqTo), 1);

                if (isCapture(m)) {
                    whiteInput->adjustInput(WHITE_INDEX(wKingSq, getCapturedPiece(m), sqTo), 0);
                    blackInput->adjustInput(BLACK_INDEX(bKingSq, getCapturedPiece(m), sqTo), 0);
                }

                return;
            } else if (mType == EN_PASSANT) {

                whiteInput->adjustInput(WHITE_INDEX(wKingSq, getCapturedPiece(m), sqTo - 8 * factor), 0);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, getCapturedPiece(m), sqTo - 8 * factor), 0);

                whiteInput->adjustInput(WHITE_INDEX(wKingSq, pFrom, sqFrom), 0);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, pFrom, sqFrom), 0);

                whiteInput->adjustInput(WHITE_INDEX(wKingSq, pFrom, sqTo), 1);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, pFrom, sqTo), 1);

                return;
            }
        } else if (pFrom % 6 == KING) {
            resetInput(b);
            return;
        }

        whiteInput->adjustInput(WHITE_INDEX(wKingSq, pFrom, sqFrom), 0);
        blackInput->adjustInput(BLACK_INDEX(bKingSq, pFrom, sqFrom), 0);

        whiteInput->adjustInput(WHITE_INDEX(wKingSq, pFrom, sqTo), 1);
        blackInput->adjustInput(BLACK_INDEX(bKingSq, pFrom, sqTo), 1);

        if (isCapture(m)) {
            whiteInput->adjustInput(WHITE_INDEX(wKingSq, getCapturedPiece(m), sqTo), 0);
            blackInput->adjustInput(BLACK_INDEX(bKingSq, getCapturedPiece(m), sqTo), 0);
        }
    }

    void onUndoMove(Board* b, Move m) {

        Square sqFrom = getSquareFrom(m);
        Square sqTo   = getSquareTo(m);
        Piece  pFrom  = getMovingPiece(m);
        Type   mType  = getType(m);
        Color  color  = pFrom / 6;
        int    factor = color == 0 ? 1 : -1;

        Square wKingSq = bitscanForward(b->getPieces(WHITE, KING));
        Square bKingSq = bitscanForward(b->getPieces(BLACK, KING));

        concat->setFlipInputs(color == BLACK);

        if (pFrom % 6 == PAWN) {

            if (isPromotion(m)) {

                // setting m_pieces
                // this->unsetPiece(sqFrom);
                whiteInput->adjustInput(WHITE_INDEX(wKingSq, pFrom, sqFrom), 1);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, pFrom, sqFrom), 1);

                // setting the piece at the target destination
                whiteInput->adjustInput(WHITE_INDEX(wKingSq, promotionPiece(m), sqTo), 0);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, promotionPiece(m), sqTo), 0);

                if (isCapture(m)) {
                    whiteInput->adjustInput(WHITE_INDEX(wKingSq, getCapturedPiece(m), sqTo), 1);
                    blackInput->adjustInput(BLACK_INDEX(bKingSq, getCapturedPiece(m), sqTo), 1);
                }

                return;
            } else if (mType == EN_PASSANT) {

                whiteInput->adjustInput(WHITE_INDEX(wKingSq, getCapturedPiece(m), sqTo - 8 * factor), 1);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, getCapturedPiece(m), sqTo - 8 * factor), 1);

                whiteInput->adjustInput(WHITE_INDEX(wKingSq, pFrom, sqFrom), 1);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, pFrom, sqFrom), 1);

                whiteInput->adjustInput(WHITE_INDEX(wKingSq, pFrom, sqTo), 0);
                blackInput->adjustInput(BLACK_INDEX(bKingSq, pFrom, sqTo), 0);

                return;
            }
        } else if (pFrom % 6 == KING) {
            resetInput(b);
            return;
        }

        whiteInput->adjustInput(WHITE_INDEX(wKingSq, pFrom, sqFrom), 1);
        blackInput->adjustInput(BLACK_INDEX(bKingSq, pFrom, sqFrom), 1);

        whiteInput->adjustInput(WHITE_INDEX(wKingSq, pFrom, sqTo), 0);
        blackInput->adjustInput(BLACK_INDEX(bKingSq, pFrom, sqTo), 0);

        if (isCapture(m)) {
            whiteInput->adjustInput(WHITE_INDEX(wKingSq, getCapturedPiece(m), sqTo), 1);
            blackInput->adjustInput(BLACK_INDEX(bKingSq, getCapturedPiece(m), sqTo), 1);
        }
    }

    Network* getNetwork() const { return network; }
    void     setNetwork(Network* network) { KingPieceNetwork::network = network; }
};

}    // namespace fecppnn

#endif