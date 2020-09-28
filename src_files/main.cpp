
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
    std::cout << worker.getNetwork() << std::endl;
    
//    net->writeNetworkStructure("net2.structure");
    
//    worker.getNetwork()->loadWeights("nn.bin");




//    Board b {"rk1Br1B1/p2n4/b7/1p6/1p1P1pp1/P4PP1/2P1N3/RNK4R b - - 0 34"};
//
//    worker.resetInput(&b);
//    worker.compute();
//
    startMeasure();
    for(int i = 0; i < 1e6; i++){
//        worker.getActiveInput()->adjustInput(128, i%2);
//        worker.getActiveInput()->adjustInput(64, i%2);
//        worker.getActiveInput()->adjustInput(17, i%2);
        worker.compute();
    }
//
    std::cout << stopMeasure() << std::endl;
//    bb_cleanUp();



    return 0;
}
