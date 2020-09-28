
#include "Bitboard.h"
#include "Board.h"
#include "Move.h"
#include "MoveOrderer.h"
#include "Tuning.h"
#include "Verification.h"
#include "uci.h"

#include <iomanip>

using namespace std;
using namespace bb;
using namespace move;

#include "fecppnn/network/Network.h"
#include "fecppnn/special/KingPieceNetwork.h"

int main(int argc, char* argv[]) {

//    if (argc == 1) {
//        uci_loop(false);
//    } else if (argc > 1 && strcmp(argv[1], "bench") == 0) {
//        uci_loop(true);
//    }

    
    bb_init();

    fecppnn::Network* net = fecppnn::createNetwork("net1.structure");
    fecppnn::KingPieceNetwork worker{net};
    
//    net->writeNetworkStructure("net2.structure");
    
//    worker.getNetwork()->loadWeights("nn.bin");



    Move m = genMove(A2,A3,QUIET,WHITE_PAWN);
    

    Board b {};

    worker.resetInput(&b);
    std::cout << worker.validateInput(&b) << std::endl;
    
    
    b.move(m);
    worker.onMove(&b, m);
    
    std::cout << worker.validateInput(&b) << std::endl;
    
    
    b.undoMove();
    worker.onUndoMove(&b, m);
    
    std::cout << worker.validateInput(&b) << std::endl;
    
    bb_cleanUp();



    return 0;
}
